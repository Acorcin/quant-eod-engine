#!/usr/bin/env python3
"""
Historical backtest loop for the live daily prediction logic.

Runs saved feature vectors through the current meta-model and evaluates
next-trading-day close-to-close PnL.
"""
import argparse
import json
import logging
from datetime import date, datetime
import numpy as np

from config.settings import PRIMARY_INSTRUMENT
from models.database import fetch_all
from models.meta_model import MetaModel
try:
    from utils.trading_calendar import next_trading_day
except Exception as exc:  # pragma: no cover - defensive fallback
    logger = logging.getLogger("backtest_loop")
    logger.warning("utils.trading_calendar import failed, using weekday fallback: %s", exc)

    def next_trading_day(run_date: date) -> date:
        from datetime import timedelta

        candidate = run_date + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest_loop")


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.fromisoformat(value).date()


def _load_price_map(instrument: str) -> dict[date, dict]:
    rows = fetch_all(
        """SELECT bar_time::date AS d, open, close
           FROM bars
           WHERE instrument = %s AND granularity = 'D' AND complete = TRUE
           ORDER BY bar_time ASC""",
        (instrument,),
    )
    return {r["d"]: {"open": float(r["open"]), "close": float(r["close"])} for r in rows}


def _resolve_next_price(price_map: dict[date, dict], d: date) -> tuple[date | None, dict | None]:
    nd = next_trading_day(d)
    guard = 0
    while nd not in price_map and guard < 30:
        nd = next_trading_day(nd)
        guard += 1
    if nd not in price_map:
        return None, None
    return nd, price_map[nd]


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    mdd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        mdd = min(mdd, dd)
    return float(mdd)


def _annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def _annualized_sortino(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / downside_std * np.sqrt(periods_per_year))


def run_backtest(instrument: str, start: date | None, end: date | None, initial_equity: float) -> dict:
    where = ["instrument = %s"]
    params: list = [instrument]
    if start:
        where.append("date >= %s")
        params.append(str(start))
    if end:
        where.append("date <= %s")
        params.append(str(end))

    fv_rows = fetch_all(
        f"""SELECT date, features
            FROM feature_vectors
            WHERE {' AND '.join(where)}
            ORDER BY date ASC""",
        tuple(params),
    )
    if not fv_rows:
        return {"error": "No feature_vectors rows found for selection"}

    price_map = _load_price_map(instrument)
    model = MetaModel()

    equity = initial_equity
    curve = []
    wins = 0
    losses = 0
    traded = 0
    prev_size = 0.0
    turnover = 0.0
    position_sizes: list[float] = []
    pnl_series: list[float] = []
    
    # Realistic Forex friction parameters
    LEVERAGE = 10.0
    SPREAD_BPS = 1.5  # 1.5 pips spread assumed
    spread_cost_mult = SPREAD_BPS / 10000.0

    for row in fv_rows:
        d = row["date"]
        features = row["features"] or {}
        pred = model.predict(features)

        nd, prices_nd = _resolve_next_price(price_map, d)
        if nd is None or prices_nd is None:
            continue

        # Real Execution logic: 
        # Feature vector is computed after day T closes.
        # We enter at the Open of T+1, exit at the Close of T+1.
        entry_price = prices_nd["open"]
        exit_price = prices_nd["close"]
        
        raw_ret = (exit_price / entry_price) - 1.0
        
        direction = pred["direction"]
        size = float(pred.get("size_multiplier", 0.0) or 0.0)

        # Apply direction, leverage, and subtract transaction costs (spread is paid on entry and exit equivalent)
        if direction == "long":
            pnl = size * LEVERAGE * (raw_ret - spread_cost_mult)
        elif direction == "short":
            pnl = size * LEVERAGE * (-raw_ret - spread_cost_mult)
        else:
            pnl = 0.0

        if direction in ("long", "short") and size > 0:
            traded += 1
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

        turnover += abs(size - prev_size)
        prev_size = size
        position_sizes.append(size)
        pnl_series.append(pnl)
        equity *= (1.0 + pnl)
        curve.append(
            {
                "date": str(d),
                "prediction_for": str(nd),
                "direction": direction,
                "probability": pred.get("probability"),
                "size": size,
                "daily_return": round(ret, 6),
                "strategy_pnl": round(pnl, 6),
                "equity": round(equity, 2),
            }
        )

    total_return = (equity / initial_equity - 1.0) if initial_equity > 0 else 0.0
    win_rate = (wins / traded) if traded > 0 else 0.0
    periods = len(pnl_series)
    years = periods / 252.0 if periods > 0 else 0.0
    cagr = ((equity / initial_equity) ** (1 / years) - 1.0) if years > 0 and initial_equity > 0 else 0.0
    pnl_arr = np.array(pnl_series, dtype=float) if pnl_series else np.array([], dtype=float)
    equity_vals = [float(x["equity"]) for x in curve]
    mdd = _max_drawdown(equity_vals)
    sharpe = _annualized_sharpe(pnl_arr)
    sortino = _annualized_sortino(pnl_arr)
    calmar = (cagr / abs(mdd)) if mdd < 0 else 0.0
    exposure = float(np.mean(np.array(position_sizes) > 0)) if position_sizes else 0.0
    avg_position_size = float(np.mean(position_sizes)) if position_sizes else 0.0
    turnover_per_day = (turnover / periods) if periods > 0 else 0.0

    performance_report = {
        "periods": periods,
        "years": round(years, 4),
        "cagr": round(cagr, 4),
        "annualized_sharpe": round(sharpe, 4),
        "annualized_sortino": round(sortino, 4),
        "max_drawdown": round(mdd, 4),
        "calmar": round(calmar, 4),
        "exposure": round(exposure, 4),
        "avg_position_size": round(avg_position_size, 4),
        "turnover_total": round(turnover, 4),
        "turnover_per_day": round(turnover_per_day, 4),
        "avg_daily_pnl": round(float(np.mean(pnl_arr)) if periods > 0 else 0.0, 6),
        "vol_daily_pnl": round(float(np.std(pnl_arr, ddof=1)) if periods > 1 else 0.0, 6),
    }

    return {
        "instrument": instrument,
        "start_date": str(fv_rows[0]["date"]),
        "end_date": str(fv_rows[-1]["date"]),
        "rows_processed": len(fv_rows),
        "trades": traded,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "initial_equity": initial_equity,
        "final_equity": round(equity, 2),
        "total_return": round(total_return, 4),
        "performance_report": performance_report,
        "equity_curve": curve,
    }


def main():
    parser = argparse.ArgumentParser(description="Historical backtest loop using stored feature vectors")
    parser.add_argument("--instrument", default=PRIMARY_INSTRUMENT, help="Instrument, e.g. EUR_USD")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--equity", type=float, default=10000.0, help="Initial equity")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    result = run_backtest(
        instrument=args.instrument,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        initial_equity=args.equity,
    )

    if "error" in result:
        logger.error(result["error"])
        raise SystemExit(1)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Backtest written to %s", args.output)
    else:
        logger.info(
            "Backtest %s %s->%s | trades=%s win_rate=%.2f%% total_return=%.2f%%",
            result["instrument"],
            result["start_date"],
            result["end_date"],
            result["trades"],
            result["win_rate"] * 100,
            result["total_return"] * 100,
        )
        report = result.get("performance_report", {})
        logger.info(
            "Perf | CAGR=%.2f%% Sharpe=%.2f Sortino=%.2f MaxDD=%.2f%% Calmar=%.2f Exposure=%.2f%% Turnover/day=%.3f",
            report.get("cagr", 0.0) * 100,
            report.get("annualized_sharpe", 0.0),
            report.get("annualized_sortino", 0.0),
            report.get("max_drawdown", 0.0) * 100,
            report.get("calmar", 0.0),
            report.get("exposure", 0.0) * 100,
            report.get("turnover_per_day", 0.0),
        )


if __name__ == "__main__":
    main()
