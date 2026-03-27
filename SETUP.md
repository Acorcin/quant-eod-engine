# Phase 1 Setup Guide — What You Need to Do

Everything below is required from **you** before the system can run.

---

## 1. OANDA Practice Account + API Token ✅ DONE

- **API Token:** `e5e5522d0c9f17de2a2053586c5a4a6c-6f956b3635f26db49d447807d1ae46ca`
- **Account ID:** `101-001-38906191-001`
- **Base URL:** `https://api-fxpractice.oanda.com` (practice/demo)

---

## 2. FRED API Key ✅ DONE

- **API Key:** `7b419f977588746a5cac272d39e03662`

---

## 3. Perplexity API Key ⏭️ SKIPPED (for now)

The system will run with a fallback: neutral sentiment score (0.0) with low confidence.
When you're ready, sign up at [docs.perplexity.ai](https://docs.perplexity.ai/) and add the key.

---

## 4. Discord Webhook (for signal delivery)

**Time:** ~2 minutes | **Cost:** Free

You need a Discord webhook URL. Two options:

### Option A: Server Channel Webhook (recommended)
1. In your Discord server, go to a channel → Edit Channel → Integrations → Webhooks
2. Click **New Webhook**
3. Name it "Quant EOD Engine"
4. Copy the **Webhook URL** (format: `https://discord.com/api/webhooks/XXXX/YYYY`)

### Option B: DM via Private Server
1. Create a personal Discord server (just for you)
2. Create a `#signals` channel
3. Set up a webhook in that channel (same steps as above)
4. Only you will see the messages

**You'll have:**
- `DISCORD_WEBHOOK_URL` — the full webhook URL

---

## 5. Create Secrets Files on VPS

SSH into your VPS and create the secrets:

```bash
cd ~/quant-eod-engine
mkdir -p secrets

# Credentials we already have
echo -n "e5e5522d0c9f17de2a2053586c5a4a6c-6f956b3635f26db49d447807d1ae46ca" > secrets/oanda_token.txt
echo -n "101-001-38906191-001" > secrets/oanda_account_id.txt
echo -n "7b419f977588746a5cac272d39e03662" > secrets/fred_key.txt

# Leave empty for now (Perplexity skipped)
echo -n "" > secrets/perplexity_key.txt

echo -n "https://discord.com/api/webhooks/1486906028296769587/dA_L8drTldVFqfbOZrR0weIepjt4kTgvFpakoQCwyhn0-G36-LMaJm51e93nX44AwH82" > secrets/discord_webhook.txt

# Set a secure database password
echo -n "a_secure_password_here" > secrets/db_password.txt

# Lock down permissions
chmod 600 secrets/*.txt
```

---

## 6. Deploy and Run

```bash
# Build and start PostgreSQL
docker compose up -d postgres

# Wait for it to be healthy (~10 sec)
docker compose ps

# Run the daily loop once to test
docker compose run --rm quant-engine

# Check logs
cat logs/daily_$(date +%Y-%m-%d).log
```

---

## 7. Set Up Cron (after testing works)

```bash
# Edit crontab
crontab -e

# Add this line (4:30 PM EST = 20:30 UTC during EDT)
# Adjust for EST (21:30 UTC) when daylight saving ends in November
30 20 * * 1-5  cd ~/quant-eod-engine && docker compose run --rm quant-engine >> /var/log/quant-cron.log 2>&1
```

---

## Checklist

- [x] OANDA practice account created
- [x] OANDA API token generated
- [x] OANDA Account ID noted
- [x] FRED account created
- [x] FRED API key generated
- [ ] Perplexity API key (skipped — add later)
- [x] Discord webhook URL created
- [ ] All secrets files created in `secrets/` directory
- [ ] `docker compose up -d postgres` succeeds
- [ ] `docker compose run --rm quant-engine` completes without critical errors
- [ ] Daily cron job configured

---

## Known Gaps (Documented, Not Blocking)

1. **German 2Y Yield:** FRED doesn't carry this series directly. System works with US 2Y alone initially.
   - Option A: ECB Statistical Data Warehouse API (free, XML-based)
   - Option B: Trading Economics API ($50/mo)

2. **Economic Calendar:** No automated feed. Perplexity AI partially covers it (once enabled).

3. **OANDA Sentiment:** ForexLabs position ratios endpoint may be deprecated. Fallback to neutral coded.

4. **Perplexity AI Sentiment:** Skipped for now. System gracefully degrades to neutral score with 0.1 confidence.
