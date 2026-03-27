# Phase 1 Setup Guide

---

## Credentials Status

| Service | Status |
|---|---|
| OANDA Token + Account ID | Done |
| FRED API Key | Done |
| Perplexity API Key | Skipped (fallback to neutral) |
| Discord Webhook | Done |

---

## Deploy on VPS

### 1. Clone the repo

```bash
git clone https://github.com/Acorcin/quant-eod-engine.git
cd quant-eod-engine
```

### 2. Create your .env file

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```env
OANDA_API_TOKEN=e5e5522d0c9f17de2a2053586c5a4a6c-6f956b3635f26db49d447807d1ae46ca
OANDA_ACCOUNT_ID=101-001-38906191-001
OANDA_BASE_URL=https://api-fxpractice.oanda.com
FRED_API_KEY=7b419f977588746a5cac272d39e03662
PERPLEXITY_API_KEY=
PERPLEXITY_MODEL=sonar-pro
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1486906028296769587/dA_L8drTldVFqfbOZrR0weIepjt4kTgvFpakoQCwyhn0-G36-LMaJm51e93nX44AwH82
DB_HOST=localhost
DB_PORT=5432
DB_NAME=quant_eod
DB_USER=postgres
DB_PASSWORD=Qu4nt_E0D_2026!
LOG_DIR=./logs
LOG_LEVEL=INFO
```

Lock it down:
```bash
chmod 600 .env
```

### 3. Start Postgres and run

```bash
docker compose up -d postgres
docker compose ps  # wait for healthy
docker compose run --rm quant-engine
```

### 4. Check results

```bash
cat logs/daily_$(date +%Y-%m-%d).log
docker exec -it quant_postgres psql -U postgres -d quant_eod -c "SELECT * FROM pipeline_runs;"
```

### 5. Set up cron (after test succeeds)

```bash
crontab -e
# 4:30 PM EST = 20:30 UTC (EDT) / 21:30 UTC (EST)
30 20 * * 1-5  cd ~/quant-eod-engine && docker compose run --rm quant-engine >> /var/log/quant-cron.log 2>&1
```

---

## Known Gaps

1. **German 2Y Yield** — FRED doesn't carry it. System runs on US 2Y alone.
2. **Economic Calendar** — Placeholder. Perplexity AI partially covers it (once enabled).
3. **OANDA Sentiment** — Endpoint may be deprecated. Fallback coded.
4. **Perplexity AI** — Skipped for now. Fallback: neutral score, 0.1 confidence.
