# NSE Under ₹300 Momentum Scanner

A self-contained, rule-based daily stock scanner for NSE stocks priced under ₹300. Outputs up to 5 high-conviction BUY candidates per run with full trade setups, scoring, and risk analysis.

## ⚠ Disclaimer

1. Educational/research tool only. **NOT SEBI-registered investment advice.**
2. Past results don't guarantee future returns. Always validate signals independently.
3. Always verify fundamentals, liquidity, and corporate actions before trading.
4. Use stop losses without exception. The stop loss IS the strategy.
5. Never risk more than 2% of capital on a single trade.
6. Authors accept no liability for any losses.

## What it does

Scans 90+ liquid NSE stocks under ₹300 using a **5-layer multi-pass shortlisting approach**:

| Layer | What it tests |
|-------|--------------|
| 1 | Price ₹50–₹300 and 20-day average volume > 100K |
| 2 | Technical momentum (RSI, MACD crossover, SMA stack, volume surge, ADX) |
| 3 | Fundamentals quick-check (P/E vs sector, D/E, ROE, revenue growth) |
| 4 | Volatility & momentum scoring (ATR, 5-day return, Bollinger position, ROC) |
| 5 | Catalyst & pattern detection (hammer, engulfing, morning star, gap-up, breakouts) |

### Output per shortlisted stock

- **Stock Info**: Symbol, Company Name, Current Price, Sector
- **Return Potential**: Estimated range (e.g., 3.5% – 7.2%)
- **Trigger Events**: Specific signals (e.g., "MACD bullish crossover + Volume surge 2.3x + Hammer candle at support")
- **Entry Level**: Recommended entry price
- **3 Targets**: Conservative (1.5:1 R:R), Moderate (2.5:1 R:R), Aggressive (3.5:1 R:R)
- **Stop Loss**: ATR-based level and % from entry
- **Scores**: Technical (0–100), Fundamental (0–100), Confidence composite
- **Risk Flags**: Warnings (high debt, low volume, near 52W high, etc.)
- **Position Sizing**: For ₹1,00,000 portfolio at 2% risk per trade
- **Validity**: 1–2 trading sessions

## Setup

Requires **Python 3.10+**.

```bash
git clone https://github.com/<you>/Top5Under300.git
cd Top5Under300
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Default scan (all 90+ stocks)
python main.py

# Custom portfolio size (₹5,00,000)
python main.py --portfolio 500000

# Narrower price band
python main.py --max-price 250 --min-price 100

# Verbose (debug) logging
python main.py --verbose

# Quick test on first 10 stocks
python main.py --limit 10
```

## Output

Three artifacts per run:

- **Console**: Rich-formatted summary table + detailed card per stock
- **`output/screener_YYYYMMDD_HHMMSS.csv`** — flat table of qualified BUYs
- **`output/screener_YYYYMMDD_HHMMSS.json`** — full structured payload

## Project structure

```
Top5Under300/
├── main.py            # Self-contained scanner — all logic inline, no src/ modules
├── email_report.py    # Email notification module (HTML formatted reports)
├── config.yaml        # All thresholds (edit to tune)
├── nse_stocks.csv     # 90+ liquid NSE stocks under ₹300
├── sector_mapping.yaml
├── requirements.txt
├── .github/workflows/
│   └── scheduled_scan.yml  # Cron schedule: Mon-Fri 9 AM IST
└── output/            # Created automatically per run
```

## Configuration (`config.yaml`)

Key thresholds you may want to tune:

| Setting | Default | Effect |
|---------|---------|--------|
| `technical.min_technical_conditions` | 4 | Require 4/5 technical conditions to pass Layer 2 |
| `technical.adx_threshold` | 25 | ADX must exceed this to count as trending |
| `technical.rsi_lower` / `rsi_upper` | 50 / 70 | RSI window for bullish zone |
| `technical.volume_surge_multiplier` | 1.5 | Volume must be this multiple of 20D average |
| `trade.atr_multiplier` | 2.0 | ATR multiplier for stop-loss distance |
| `scoring.max_buy_signals` | 5 | Maximum stocks shown in output |

If the scanner returns 0 BUYs most days, that is **expected behavior** — strict confluence protects capital. To relax:
- Reduce `technical.min_technical_conditions` from 4 to 3
- Reduce `technical.adx_threshold` from 25 to 20

## Dependencies

All free, no paid APIs required:

- `yfinance` — price history and fundamentals
- `ta` — RSI, MACD, ADX, Bollinger Bands, ATR, ROC
- `pandas` / `numpy` — calculations
- `rich` — colored console output
- `pyyaml` — configuration loading

## Automated Scheduling & Email Notifications

The screener can run automatically **Monday–Friday at 9:00 AM IST** (before market opens) via GitHub Actions and send results to your email.

### Setup

1. **Configure GitHub Secrets** in your repository (`Settings → Secrets and variables → Actions`):

   | Secret | Description | Example |
   |--------|-------------|---------|
   | `EMAIL_SENDER` | Sender Gmail address | `yourname@gmail.com` |
   | `EMAIL_PASSWORD` | Gmail App Password (not your regular password) | `abcd efgh ijkl mnop` |
   | `EMAIL_RECIPIENTS` | Comma-separated recipient emails | `you@gmail.com,friend@gmail.com` |
   | `SMTP_HOST` | SMTP server (optional, default: `smtp.gmail.com`) | `smtp.gmail.com` |
   | `SMTP_PORT` | SMTP port (optional, default: `587`) | `587` |

2. **Generate a Gmail App Password**:
   - Go to [Google Account → Security → App Passwords](https://myaccount.google.com/apppasswords)
   - Select "Mail" and generate a 16-character app password
   - Use this as `EMAIL_PASSWORD` (not your regular Gmail password)

3. **Enable the workflow**: The workflow at `.github/workflows/scheduled_scan.yml` runs automatically on schedule. You can also trigger it manually from the Actions tab.

### Manual trigger

You can test the workflow anytime from GitHub → Actions → "Scheduled Stock Screener" → "Run workflow".

### Email report

The email includes:
- Summary table of all shortlisted stocks
- Detailed cards for each stock with entry, targets, stop loss, scores, and risk flags
- Position sizing recommendations
- Formatted HTML for easy reading on any device

### Run email locally (for testing)

```bash
# Set environment variables
export EMAIL_SENDER="your@gmail.com"
export EMAIL_PASSWORD="your-app-password"
export EMAIL_RECIPIENTS="recipient@gmail.com"

# Run screener first
python main.py

# Send email with latest results
python email_report.py
```

## Known limitations

- **yfinance fundamentals can be stale or missing** for smaller Indian stocks. Always verify on Screener.in or company filings.
- **No FII/DII/OI modeling.** Institutional flows not considered.
- **No automated trading.** Output is research candidates only.
