# NSE Stock Screener

A rule-based, LLM-free, fully deterministic stock screener for the National Stock Exchange of India (NSE). Outputs 0–5 qualified BUY signals per day with complete trade setups: entry, stop loss, target levels, position sizing, and validity.

## ⚠ Disclaimer

1. Screening/research tool for educational purposes only. **NOT SEBI-registered investment advice.**
2. Past results don't guarantee future returns. Expected hit rate: 45–55%.
3. Always verify fundamentals, news, and corporate actions independently.
4. Use stop losses without exception. The stop loss IS the strategy.
5. Never risk more than 2% of capital on a single trade.
6. Backtest before deploying real capital.
7. Authors accept no liability for any losses.
8. No screener reliably predicts short-term price moves. Treat signals as research candidates only.

## What it does

For each NSE stock priced between ₹50 and ₹300, this tool applies a 4-layer confluence test:

1. **Technical** — ≥4 of 6 must hold (RSI dual-state, MACD bullish crossover, SMA stack, volume surge, Supertrend, ADX > 25).
2. **Fundamental** — PE below sector median × 1.2, D/E < 1.5, ROE > 10%, positive revenue growth.
3. **News sentiment** — RSS-based regex matching with word-boundary regex, negation handling, recency weighting, and fuzzy deduplication. No LLM.
4. **Sector tailwind** — Stock's sector outperforming NIFTY 50 over 20 trading days.

Stocks passing all 4 layers receive a trade setup with entry, ATR-based stop loss, two R:R targets, and position sizing capped at 20% of portfolio per trade.

## What it deliberately doesn't do

- No LLM, ML, or AI calls anywhere. Fully deterministic.
- No paid APIs. Free data sources only (yfinance, NSE public endpoints, RSS).
- No claim of predictive power. The 4-layer test is a research filter.
- No automated trading. Output is research candidates only.

## Setup

Requires **Python 3.10+**.

```bash
git clone https://github.com/<you>/nse-screener.git
cd nse-screener
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Default scan
python main.py

# Custom portfolio size (₹5,00,000)
python main.py --portfolio 500000

# Narrower price band
python main.py --max-price 250 --min-price 100

# Skip trade setups (just count qualifying stocks)
python main.py --dry-run

# Skip news layer (sentiment treated as neutral)
python main.py --no-news

# Verbose logging
python main.py --verbose

# Limit universe for quick testing
python main.py --limit 20
```

## Output

Three files per run, plus rich console display:

- `output/screener_YYYYMMDD_HHMMSS.csv` — flat table of qualified BUYs
- `output/screener_YYYYMMDD_HHMMSS.json` — full structured payload, schema_version "3.0"
- `logs/universe_YYYYMMDD_HHMMSS.log` — per-stock pass/fail log for audit

Console display includes a top-N table plus a detailed card per BUY with the technical checklist, fundamentals, trade setup, position sizing, catalysts found, sector context, risk flags, and validity expiry.

## Project structure

```
nse_screener/
├── main.py                # CLI orchestration
├── config.yaml            # All thresholds (no magic numbers in code)
├── sector_mapping.yaml    # yfinance sector → NIFTY sector index
├── requirements.txt
├── data/
│   ├── nse_stocks.csv     # Static fallback universe (~20 sample stocks)
│   └── asm_gsm_list.csv   # Cached ASM/GSM surveillance list
├── src/
│   ├── indicators.py      # RSI, MACD, SMA, ATR, ADX, Supertrend
│   ├── filters.py         # 8 hard filters
│   ├── confluence.py      # 4-layer BUY qualification
│   ├── news_sentiment.py  # RSS + word-boundary regex + negation + dedup
│   ├── trade_setup.py     # Entry / SL / Targets / position sizing
│   ├── risk_flags.py      # 15+ deterministic risk warnings
│   ├── scorer.py          # 4-component composite score
│   ├── data_fetcher.py    # yfinance + NSE APIs + RSS, with diskcache
│   ├── reporter.py        # Console (rich) + CSV + JSON output
│   └── calendar_utils.py  # NSE trading-day arithmetic
├── tests/                 # pytest suite (64 tests)
└── output/, logs/, .cache/
```

## Architecture decisions

- **LLM-first classification rejected.** All decisions are rule-based and auditable.
- **Confidence-scored cross-page logic** not needed — RSS items are flat documents.
- **No survivorship bias in sector PE.** Sector median is computed from all stocks under ₹300, not the filtered universe.
- **Crypto-shredding / GDPR.** Not applicable; only public market data is processed.
- **Two-stage PII detection** not applicable; this tool processes no PII.

## Tuning

If the screener returns 0 BUYs on most trading days, that is **expected behavior**. Strict confluence protects capital. To loosen:

- `config.yaml → technical.min_technical_conditions`: reduce from 4 to 3.
- `config.yaml → technical.adx_threshold`: reduce from 25 to 20.
- `config.yaml → sentiment.min_weighted_catalyst_count`: reduce from 1.0 to 0.5.

Don't loosen if you don't understand the trade-off (more false positives).

## Known limitations

- **yfinance fundamentals can be stale or wrong** for less-followed Indian stocks. Always verify on screener.in or the company's filings.
- **RSS-only sentiment is keyword-based**, not NLP. It catches strong signals (SEBI action, earnings beats) but misses nuance.
- **Sector mapping is approximate.** yfinance's sector strings don't map cleanly to NIFTY sector indices; we use the best available match and fall back to NIFTY 50.
- **No FII/DII/OI modeling.** Institutional flows and options data are not considered.
- **NSE APIs are unreliable from datacenter IPs.** The tool handles this with graceful fallback to the static universe CSV and cached ASM lists, but expect degraded data from cloud VMs.

## Testing

```bash
pytest tests/ -v
```

All 64 tests should pass. The suite covers indicators, filters, trade setup math, sentiment rules (word boundaries, negation, dedup, recency), scoring, and risk flags.

## License

For personal research use. Not for redistribution as financial advice.
