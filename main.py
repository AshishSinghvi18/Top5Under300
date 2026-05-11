"""NSE Stock Screener — orchestration entry point.

Loads config, fetches data, applies filters and confluence layers,
computes trade setups and scores, ranks and emits output.

Run:
    python main.py
    python main.py --portfolio 500000
    python main.py --max-price 250 --min-price 100
    python main.py --dry-run
    python main.py --no-news
    python main.py --verbose
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger
from rich.console import Console

from src.calendar_utils import latest_trading_day
from src.confluence import (
    Layer1Result,
    Layer2Result,
    Layer4Result,
    compute_20d_return,
    evaluate_layer1_technical,
    evaluate_layer2_fundamentals,
    evaluate_layer4_sector,
)
from src.data_fetcher import (
    all_rss_failed,
    fetch_asm_gsm_list,
    fetch_corporate_actions,
    fetch_fundamentals,
    fetch_ohlcv,
    fetch_promoter_holding,
    fetch_rss_news,
    fetch_sector_index_returns,
    load_nse_universe,
)
from src.filters import apply_hard_filters
from src.news_sentiment import evaluate_sentiment
from src.reporter import (
    BuyResult,
    print_detail_card,
    print_disclaimer,
    print_main_table,
    print_summary,
    print_zero_results,
    write_csv,
    write_json,
    write_universe_log,
)
from src.risk_flags import generate_risk_flags
from src.scorer import compute_all_scores, rank_top_n
from src.trade_setup import compute_trade_setup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE Stock Screener — rule-based BUY signals.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--sector-mapping", default="sector_mapping.yaml")
    p.add_argument("--universe-csv", default="data/nse_stocks.csv")
    p.add_argument("--portfolio", type=float, default=None, help="Override portfolio size (INR)")
    p.add_argument("--max-price", type=float, default=None)
    p.add_argument("--min-price", type=float, default=None)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--dry-run", action="store_true", help="Run pipeline without trade setups")
    p.add_argument("--no-news", action="store_true", help="Skip news layer (treat as neutral)")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    p.add_argument("--limit", type=int, default=None, help="Limit universe to first N stocks (for testing)")
    return p.parse_args()


def configure_logging(verbose: bool, logs_dir: str) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<level>{level: <7}</level> | {message}")
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"screener_{datetime.now().strftime('%Y%m%d')}.log"
    logger.add(str(log_file), level="DEBUG", rotation="10 MB", retention="14 days")


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_sector_mapping(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def map_sector_to_index(yf_sector: Optional[str], sector_mapping: Dict[str, Any]) -> str:
    """Look up the NIFTY sector index ticker for a yfinance sector string."""
    if not yf_sector:
        return sector_mapping["default_index"]
    return sector_mapping["sector_to_index"].get(yf_sector, sector_mapping["default_index"])


def compute_sector_median_pe(
    sector_pe_pairs: List[tuple],
    sector_mapping: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Group (sector_index, pe) pairs by index, compute median per sector.

    Sectors with fewer than min_sample_size stocks use static fallback.
    """
    min_size = int(config["sector_pe_dynamic"]["min_sample_size"])
    fallback = config["sector_pe_fallback"]
    by_index: Dict[str, List[float]] = {}
    for idx, pe in sector_pe_pairs:
        if pe is None or pd.isna(pe) or pe <= 0:
            continue
        by_index.setdefault(idx, []).append(pe)

    medians: Dict[str, float] = {}
    for idx, label in sector_mapping["index_to_pe_key"].items():
        pes = by_index.get(idx, [])
        if len(pes) >= min_size:
            medians[idx] = float(statistics.median(pes))
        else:
            medians[idx] = float(fallback.get(label, fallback.get("Default", 22)))
    return medians


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.max_price is not None:
        config["screener"]["max_price"] = args.max_price
    if args.min_price is not None:
        config["screener"]["min_price"] = args.min_price
    portfolio_size = args.portfolio if args.portfolio is not None else config["portfolio"]["default_size"]

    configure_logging(args.verbose, args.logs_dir)
    console = Console()

    start_time = time.time()
    run_ts = datetime.now()
    print_disclaimer(console)

    sector_mapping = load_sector_mapping(args.sector_mapping)

    logger.info("Loading NSE universe...")
    universe = load_nse_universe(config, static_csv=args.universe_csv)
    if args.limit:
        universe = universe.head(args.limit)
    console.print(f"[cyan]Universe size:[/cyan] {len(universe)} stocks")

    logger.info("Fetching ASM/GSM list (best-effort)...")
    asm_gsm_set = fetch_asm_gsm_list(config)

    logger.info("Fetching sector index returns...")
    sector_returns = fetch_sector_index_returns(args.sector_mapping, config)
    nifty_default = sector_mapping["default_index"]
    nifty_20d = sector_returns.get(nifty_default, float("nan"))

    if args.no_news:
        logger.info("--no-news: skipping news fetch")
        news_items: List = []
        sources_failed = True
    else:
        logger.info("Fetching news from RSS feeds...")
        news_items = fetch_rss_news(config)
        sources_failed = all_rss_failed(news_items)

    # Two-pass over universe.
    # Pass 1: gather (sector_index, pe) for all stocks under max_price for median computation.
    logger.info("Pass 1: gathering sector PE samples...")
    sector_pe_pairs: List[tuple] = []
    stock_data_cache: Dict[str, Dict[str, Any]] = {}

    for idx, row in universe.iterrows():
        symbol = row["Symbol"]
        df = fetch_ohlcv(symbol, config)
        if df is None:
            continue
        latest_close = float(df["Close"].iloc[-1])
        if latest_close >= config["screener"]["max_price"] or latest_close < config["screener"]["min_price"]:
            continue
        fund = fetch_fundamentals(symbol, config)
        if fund is None:
            continue
        sector_idx = map_sector_to_index(fund.get("sector"), sector_mapping)
        stock_data_cache[symbol] = {
            "df": df,
            "fund": fund,
            "sector_idx": sector_idx,
            "series": row.get("Series", "EQ"),
        }
        if fund.get("pe") is not None:
            sector_pe_pairs.append((sector_idx, fund["pe"]))

    logger.info(f"Pass 1: collected {len(stock_data_cache)} candidate stocks, {len(sector_pe_pairs)} PE samples")
    sector_medians = compute_sector_median_pe(sector_pe_pairs, sector_mapping, config)
    logger.debug(f"Sector medians: {sector_medians}")

    # Pass 2: full evaluation
    logger.info("Pass 2: applying hard filters and confluence layers...")
    qualified: List[BuyResult] = []
    universe_log: List[Dict[str, Any]] = []
    passed_hard = 0

    for symbol, data in stock_data_cache.items():
        df = data["df"]
        fund = data["fund"]
        sector_idx = data["sector_idx"]
        series = data["series"]

        # Best-effort: corporate actions
        corp_actions = fetch_corporate_actions(symbol, config, config["screener"]["corporate_action_lookahead_days"])
        corp_action_skipped = corp_actions is None

        filt = apply_hard_filters(
            symbol=symbol,
            df=df,
            fundamentals=fund,
            series_code=series,
            asm_gsm_set=asm_gsm_set,
            corp_actions=corp_actions,
            config=config,
        )
        if not filt.passed:
            universe_log.append({"symbol": symbol, "stage": "hard_filter", "passed": False, "reason": filt.rejection_reason})
            continue
        passed_hard += 1
        asm_check_skipped = asm_gsm_set is None

        # Best-effort: promoter holding
        promoter = fetch_promoter_holding(symbol, config)
        promoter_skipped = promoter is None

        # Layer 1
        sec_pe = sector_medians.get(sector_idx, config["sector_pe_fallback"].get("Default", 22))
        l1 = evaluate_layer1_technical(df, config)
        l2 = evaluate_layer2_fundamentals(fund, promoter, sec_pe, config)

        company_name = fund.get("long_name") or symbol
        if args.no_news or sources_failed:
            l3 = evaluate_sentiment(symbol, company_name, [], config, all_sources_failed=True)
        else:
            l3 = evaluate_sentiment(symbol, company_name, news_items, config, all_sources_failed=False)

        stock_20d = compute_20d_return(df["Close"])
        l4 = evaluate_layer4_sector(stock_20d, sector_idx, sector_returns, nifty_default=nifty_default)

        all_passed = l1.passed and l2.passed and l3.passed and l4.passed

        universe_log.append({
            "symbol": symbol,
            "stage": "qualified" if all_passed else "rejected_layers",
            "layer1_passed": l1.passed,
            "layer1_conditions_met": l1.conditions_met,
            "layer2_passed": l2.passed,
            "layer3_passed": l3.passed,
            "layer3_neg_matches": len(l3.negative_matches),
            "layer3_weighted_count": l3.weighted_count,
            "layer4_passed": l4.passed,
        })

        if not all_passed:
            continue

        # Dry-run skips trade setups but still reports the qualify.
        if args.dry_run:
            console.print(f"[green]✓[/green] {symbol} qualified ({l1.conditions_met}/6 technical)")
            continue

        setup = compute_trade_setup(df, config, portfolio_size=portfolio_size)
        if setup is None:
            universe_log[-1]["stage"] = "rejected_trade_setup"
            continue

        scores = compute_all_scores(fund, l1, l2, l3, l4, df, stock_20d, config)

        flags = generate_risk_flags(
            df=df,
            fundamentals=fund,
            layer1=l1,
            layer2=l2,
            layer3=l3,
            layer4=l4,
            setup=setup,
            promoter_skipped=promoter_skipped,
            corp_action_skipped=corp_action_skipped,
            asm_check_skipped=asm_check_skipped,
            config=config,
        )

        qualified.append(BuyResult(
            symbol=symbol,
            company=company_name,
            layer1=l1,
            layer2=l2,
            layer3=l3,
            layer4=l4,
            setup=setup,
            scores=scores,
            fundamentals=fund,
            stock_20d_return=stock_20d,
            risk_flags=flags,
            warnings=filt.warnings,
        ))

    # Rank
    top = rank_top_n(qualified, n=config["scoring"]["max_buy_signals"])

    stats: Dict[str, Any] = {
        "run_timestamp": run_ts.isoformat(),
        "trading_day": latest_trading_day(run_ts).isoformat(),
        "scanned_count": len(universe),
        "passed_hard_filters_count": passed_hard,
        "qualified_buys_count": len(top),
        "nifty_20d_return": nifty_20d if not pd.isna(nifty_20d) else None,
    }

    print_summary(console, stats)

    if args.dry_run:
        console.print(f"\n[dim]Dry run complete. {len(qualified)} stocks qualified for trade setup.[/dim]")
        return 0

    if not top:
        print_zero_results(console)
    else:
        print_main_table(console, top)
        console.print()
        for r in top:
            print_detail_card(console, r)

    # File outputs
    ts_str = run_ts.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    csv_path = out_dir / f"screener_{ts_str}.csv"
    json_path = out_dir / f"screener_{ts_str}.json"
    log_path = Path(args.logs_dir) / f"universe_{ts_str}.log"

    write_csv(top, str(csv_path))
    write_json(top, stats, str(json_path))
    write_universe_log(universe_log, str(log_path))

    elapsed = time.time() - start_time
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
