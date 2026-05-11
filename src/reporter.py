"""Reporter module — console display (rich), CSV, and JSON output."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class BuyResult:
    """Holds all data for a qualified stock pick."""

    symbol: str
    company: str
    layer1: Any  # Layer1Result
    layer2: Any  # Layer2Result
    layer3: Any  # SentimentResult
    layer4: Any  # Layer4Result
    setup: Any  # TradeSetup
    scores: Dict[str, float] = field(default_factory=dict)
    fundamentals: Dict[str, Any] = field(default_factory=dict)
    stock_20d_return: float = 0.0
    risk_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------


def print_disclaimer(console: Console) -> None:
    """Print a disclaimer warning box."""
    text = (
        "[bold red]DISCLAIMER[/bold red]\n\n"
        "This screener is for [bold]educational and informational purposes only[/bold].\n"
        "It does NOT constitute financial advice. Always do your own research\n"
        "and consult a SEBI-registered advisor before making investment decisions.\n"
        "Past performance does not guarantee future results."
    )
    console.print(Panel(text, border_style="red", title="⚠ Warning", expand=False))


def print_summary(console: Console, stats: Dict[str, Any]) -> None:
    """Print run summary statistics."""
    console.print()
    console.print(Panel.fit(
        f"[cyan]Timestamp:[/cyan]  {stats.get('run_timestamp', 'N/A')}\n"
        f"[cyan]Trading Day:[/cyan] {stats.get('trading_day', 'N/A')}\n"
        f"[cyan]Scanned:[/cyan]     {stats.get('scanned_count', 0)} stocks\n"
        f"[cyan]Passed Filters:[/cyan] {stats.get('passed_hard_filters_count', 0)}\n"
        f"[cyan]Qualified:[/cyan]   {stats.get('qualified_buys_count', 0)}\n"
        f"[cyan]NIFTY 20d Return:[/cyan] "
        f"{_fmt_pct(stats.get('nifty_20d_return'))}",
        title="📊 Run Summary",
        border_style="blue",
    ))


def print_zero_results(console: Console) -> None:
    """Print message when no stocks qualified."""
    console.print(
        Panel(
            "[yellow]No stocks qualified as buy signals today.[/yellow]\n"
            "Try adjusting filters or check market conditions.",
            title="0 Results",
            border_style="yellow",
            expand=False,
        )
    )


def print_main_table(console: Console, results: List[BuyResult]) -> None:
    """Print a ranked summary table of top picks."""
    table = Table(title="🏆 Top Buy Signals", show_lines=True)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Symbol", style="green")
    table.add_column("Company")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("Entry", justify="right")
    table.add_column("SL", justify="right")
    table.add_column("T1", justify="right")
    table.add_column("T2", justify="right")
    table.add_column("R:R", justify="right")
    table.add_column("Qty", justify="right")

    for rank, r in enumerate(results, 1):
        s = r.setup
        table.add_row(
            str(rank),
            r.symbol,
            _trunc(r.company, 22),
            f"{r.scores.get('composite_score', 0):.1f}",
            f"₹{s.entry:.2f}" if s else "—",
            f"₹{s.stop_loss:.2f}" if s else "—",
            f"₹{s.target1:.2f}" if s else "—",
            f"₹{s.target2:.2f}" if s else "—",
            f"1:{s.rr1:.1f}" if s else "—",
            str(s.position_size) if s else "—",
        )

    console.print(table)


def print_detail_card(console: Console, result: BuyResult) -> None:
    """Print a detailed card for a single stock."""
    r = result
    s = r.setup
    lines: List[str] = []

    # Technical checklist (Layer 1)
    lines.append("[bold underline]Technical Checklist[/bold underline]")
    if r.layer1:
        details = getattr(r.layer1, "details", {}) or {}
        met = getattr(r.layer1, "conditions_met", "?")
        lines.append(f"  Conditions met: {met}/6")
        for key, val in details.items():
            icon = "[green]✓[/green]" if val else "[red]✗[/red]"
            lines.append(f"  {icon} {key}")
    lines.append("")

    # Fundamentals (Layer 2)
    lines.append("[bold underline]Fundamentals[/bold underline]")
    fund = r.fundamentals or {}
    for k in ("pe", "debt_to_equity", "roe", "revenue_growth", "market_cap", "sector"):
        val = fund.get(k)
        lines.append(f"  {k}: {_fmt_val(val)}")
    lines.append("")

    # Sentiment (Layer 3)
    lines.append("[bold underline]Sentiment[/bold underline]")
    if r.layer3:
        lines.append(f"  Label: {getattr(r.layer3, 'label', '?')}")
        pos = getattr(r.layer3, "positive_matches", [])
        neg = getattr(r.layer3, "negative_matches", [])
        lines.append(f"  Positive catalysts: {len(pos)}")
        lines.append(f"  Negative signals:   {len(neg)}")
    lines.append("")

    # Sector context (Layer 4)
    lines.append("[bold underline]Sector Context[/bold underline]")
    if r.layer4:
        lines.append(f"  Sector 20d return: {_fmt_pct(getattr(r.layer4, 'sector_return', None))}")
        lines.append(f"  NIFTY 20d return:  {_fmt_pct(getattr(r.layer4, 'nifty_return', None))}")
        lines.append(f"  Stock 20d return:  {_fmt_pct(r.stock_20d_return)}")
    lines.append("")

    # Trade setup
    if s:
        lines.append("[bold underline]Trade Setup[/bold underline]")
        lines.append(f"  Entry:  ₹{s.entry:.2f}")
        lines.append(f"  SL:     ₹{s.stop_loss:.2f} ({s.sl_percent:.2f}%)")
        lines.append(f"  T1:     ₹{s.target1:.2f}  (R:R 1:{s.rr1:.1f})")
        lines.append(f"  T2:     ₹{s.target2:.2f}  (R:R 1:{s.rr2:.1f})")
        lines.append(f"  ATR:    ₹{s.atr:.2f}")
        lines.append("")
        lines.append("[bold underline]Position Sizing[/bold underline]")
        lines.append(f"  Qty:    {s.position_size}")
        lines.append(f"  Value:  ₹{s.position_value:,.2f}")
        lines.append(f"  Risk:   ₹{s.risk_amount:,.2f}")
        lines.append(f"  Validity: {s.validity_days} trading days")
        lines.append("")

    # Scores
    lines.append("[bold underline]Scores[/bold underline]")
    for k, v in r.scores.items():
        lines.append(f"  {k}: {v:.2f}")
    lines.append("")

    # Risk flags
    if r.risk_flags:
        lines.append("[bold underline]Risk Flags[/bold underline]")
        for f in r.risk_flags:
            lines.append(f"  [yellow]{f}[/yellow]")
        lines.append("")

    # Warnings
    if r.warnings:
        lines.append("[bold underline]Warnings[/bold underline]")
        for w in r.warnings:
            lines.append(f"  [dim]{w}[/dim]")

    console.print(Panel(
        "\n".join(lines),
        title=f"[bold green]{r.symbol}[/bold green] — {r.company}",
        border_style="green",
        expand=False,
    ))


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def write_csv(results: List[BuyResult], path: str) -> None:
    """Write results to CSV, creating parent directories as needed."""
    _ensure_parent(path)
    fieldnames = [
        "rank", "symbol", "company", "composite_score",
        "technical_score", "fundamental_score", "sentiment_score", "sector_score",
        "entry", "stop_loss", "sl_percent", "target1", "target2",
        "rr1", "rr2", "position_size", "position_value", "risk_amount",
        "atr", "validity_days", "stock_20d_return",
        "pe", "debt_to_equity", "roe", "revenue_growth", "market_cap", "sector",
        "sentiment_label", "risk_flags", "warnings",
    ]
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for rank, r in enumerate(results, 1):
                s = r.setup
                fund = r.fundamentals or {}
                row = {
                    "rank": rank,
                    "symbol": r.symbol,
                    "company": r.company,
                    "composite_score": r.scores.get("composite_score", ""),
                    "technical_score": r.scores.get("technical_score", ""),
                    "fundamental_score": r.scores.get("fundamental_score", ""),
                    "sentiment_score": r.scores.get("sentiment_score", ""),
                    "sector_score": r.scores.get("sector_score", ""),
                    "entry": s.entry if s else "",
                    "stop_loss": s.stop_loss if s else "",
                    "sl_percent": s.sl_percent if s else "",
                    "target1": s.target1 if s else "",
                    "target2": s.target2 if s else "",
                    "rr1": s.rr1 if s else "",
                    "rr2": s.rr2 if s else "",
                    "position_size": s.position_size if s else "",
                    "position_value": s.position_value if s else "",
                    "risk_amount": s.risk_amount if s else "",
                    "atr": s.atr if s else "",
                    "validity_days": s.validity_days if s else "",
                    "stock_20d_return": r.stock_20d_return,
                    "pe": fund.get("pe", ""),
                    "debt_to_equity": fund.get("debt_to_equity", ""),
                    "roe": fund.get("roe", ""),
                    "revenue_growth": fund.get("revenue_growth", ""),
                    "market_cap": fund.get("market_cap", ""),
                    "sector": fund.get("sector", ""),
                    "sentiment_label": getattr(r.layer3, "label", "") if r.layer3 else "",
                    "risk_flags": "; ".join(r.risk_flags) if r.risk_flags else "",
                    "warnings": "; ".join(r.warnings) if r.warnings else "",
                }
                writer.writerow(row)
        logger.info("CSV written to {}", path)
    except OSError:
        logger.exception("Failed to write CSV to {}", path)


def write_json(results: List[BuyResult], stats: Dict[str, Any], path: str) -> None:
    """Write results and stats to JSON with schema_version 3.0."""
    _ensure_parent(path)
    payload = {
        "schema_version": "3.0",
        "stats": stats,
        "results": [_buy_result_to_dict(r, rank) for rank, r in enumerate(results, 1)],
    }
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("JSON written to {}", path)
    except OSError:
        logger.exception("Failed to write JSON to {}", path)


def write_universe_log(log: List[Dict[str, Any]], path: str) -> None:
    """Write universe processing log to a JSON-lines file."""
    _ensure_parent(path)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            for entry in log:
                fh.write(json.dumps(entry, default=str) + "\n")
        logger.info("Universe log written to {} ({} entries)", path, len(log))
    except OSError:
        logger.exception("Failed to write universe log to {}", path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _trunc(text: str, length: int) -> str:
    return text if len(text) <= length else text[: length - 1] + "…"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _fmt_val(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _safe_asdict(obj: Any) -> Any:
    """Convert dataclass to dict; fall back to str for non-serialisable objects."""
    if obj is None:
        return None
    try:
        return asdict(obj)
    except (TypeError, AttributeError):
        return str(obj)


def _buy_result_to_dict(r: BuyResult, rank: int) -> Dict[str, Any]:
    """Convert a BuyResult to a JSON-friendly dictionary."""
    return {
        "rank": rank,
        "symbol": r.symbol,
        "company": r.company,
        "scores": r.scores,
        "fundamentals": r.fundamentals,
        "stock_20d_return": r.stock_20d_return,
        "risk_flags": r.risk_flags,
        "warnings": r.warnings,
        "layer1": _safe_asdict(r.layer1),
        "layer2": _safe_asdict(r.layer2),
        "layer3": _safe_asdict(r.layer3),
        "layer4": _safe_asdict(r.layer4),
        "setup": _safe_asdict(r.setup),
    }
