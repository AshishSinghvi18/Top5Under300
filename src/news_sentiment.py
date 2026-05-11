"""Layer 3 — RSS-based keyword sentiment analysis.

Matches stock symbol / company name in news titles and summaries using
word-boundary regex, classifies hits as positive or negative with negation
handling, applies recency weighting, deduplicates with rapidfuzz, and
returns a pass/fail verdict.

No LLM is used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from rapidfuzz import fuzz as _fuzz

    def _fuzz_ratio(a: str, b: str) -> float:
        return _fuzz.ratio(a, b)

except ImportError:  # pragma: no cover
    logger.warning("rapidfuzz not installed – fuzzy dedup disabled")

    def _fuzz_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        return 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    passed: bool
    label: str  # "positive", "neutral", or "negative"
    positive_matches: List[Dict[str, Any]] = field(default_factory=list)
    negative_matches: List[Dict[str, Any]] = field(default_factory=list)
    weighted_count: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

NEGATIVE_KEYWORDS: List[str] = [
    "fraud",
    "scam",
    "penalty",
    "SEBI action",
    "downgrade",
    "default",
    "loss",
    "probe",
    "investigation",
    "ban",
    "suspension",
    "FIR",
    "arrest",
    "money laundering",
    "insolvency",
    "NCLT",
    "delisting warning",
]

POSITIVE_KEYWORDS: List[str] = [
    "upgrade",
    "outperform",
    "earnings beat",
    "revenue growth",
    "order win",
    "expansion",
    "acquisition",
    "partnership",
    "approval",
    "contract",
    "record high",
    "strong results",
    "beat estimates",
    "dividend",
]

NEGATION_WORDS: List[str] = ["no", "not", "neither", "denied", "unlikely", "without"]

# Pre-compiled patterns (built once at import time)
_NEG_PATTERNS = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in NEGATIVE_KEYWORDS]
_POS_PATTERNS = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in POSITIVE_KEYWORDS]
_NEGATION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in NEGATION_WORDS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_entity_pattern(symbol: str, company_name: Optional[str]) -> re.Pattern[str]:
    """Build a compiled regex that matches the stock symbol or company name."""
    parts = [re.escape(symbol)]
    if company_name and company_name != symbol:
        parts.append(re.escape(company_name))
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def _parse_published(published: Any) -> Optional[datetime]:
    """Normalise *published* to a timezone-aware UTC datetime."""
    if published is None:
        return None
    if isinstance(published, datetime):
        if published.tzinfo is None:
            return published.replace(tzinfo=timezone.utc)
        return published.astimezone(timezone.utc)
    if isinstance(published, str):
        # Try fromisoformat first (handles microseconds and timezone offsets)
        try:
            dt = datetime.fromisoformat(published)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
        ):
            try:
                dt = datetime.strptime(published, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
        logger.debug(f"Could not parse published date: {published!r}")
        return None
    return None


def _days_ago(dt: datetime, now: datetime) -> float:
    return max((now - dt).total_seconds() / 86400, 0.0)


def _recency_weight(days: float, weights: Dict[str, float]) -> float:
    if days <= 3:
        return weights.get("near", 1.0)
    if days <= 7:
        return weights.get("medium", 0.7)
    return weights.get("far", 0.4)


def _is_negated(text: str, match_start: int, window: int = 40) -> bool:
    """Return True if a negation word appears within *window* chars before the match."""
    start = max(0, match_start - window)
    preceding = text[start:match_start]
    return bool(_NEGATION_PATTERN.search(preceding))


def _text_for_item(item: Dict[str, Any]) -> str:
    title = item.get("title") or ""
    summary = item.get("summary") or ""
    return f"{title} {summary}"


def _dedup(matches: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """Remove near-duplicate matches based on fuzzy title similarity."""
    if not matches or threshold <= 0:
        return matches
    kept: List[Dict[str, Any]] = []
    for m in matches:
        title = m.get("title", "")
        if any(_fuzz_ratio(title, k.get("title", "")) >= threshold for k in kept):
            continue
        kept.append(m)
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_sentiment(
    symbol: str,
    company_name: Optional[str],
    news_items: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    all_sources_failed: bool = False,
) -> SentimentResult:
    """Evaluate Layer-3 news sentiment for a single stock.

    Parameters
    ----------
    symbol:
        NSE ticker (e.g. ``"RELIANCE"``).
    company_name:
        Human-readable company name for matching.
    news_items:
        Aggregated RSS news items across all feeds.
    config:
        Full screener config dict (``sentiment`` section used).
    all_sources_failed:
        If ``True`` every RSS feed failed — give benefit of doubt.

    Returns
    -------
    SentimentResult
    """
    cfg = config.get("sentiment", {})
    neg_days = cfg.get("negative_lookback_days", 7)
    pos_days = cfg.get("positive_lookback_days", 14)
    weights = cfg.get("recency_weights", {"near": 1.0, "medium": 0.7, "far": 0.4})
    min_catalyst = cfg.get("min_weighted_catalyst_count", 1.0)
    dedup_thresh = cfg.get("dedup_fuzzy_threshold", 85)

    now = datetime.now(tz=timezone.utc)
    entity_re = _build_entity_pattern(symbol, company_name)

    positive_matches: List[Dict[str, Any]] = []
    negative_matches: List[Dict[str, Any]] = []

    for item in news_items:
        if item.get("fetch_failed"):
            continue

        text = _text_for_item(item)
        if not entity_re.search(text):
            continue

        pub = _parse_published(item.get("published"))
        days = _days_ago(pub, now) if pub else 0.0
        weight = _recency_weight(days, weights)

        # --- negative scan (within negative lookback) ---
        if pub is None or days <= neg_days:
            for pat, kw in zip(_NEG_PATTERNS, NEGATIVE_KEYWORDS):
                m = pat.search(text)
                if m and not _is_negated(text, m.start()):
                    negative_matches.append({
                        "title": item.get("title", ""),
                        "source": item.get("source", ""),
                        "keyword": kw,
                        "published": str(pub) if pub else None,
                        "weight": weight,
                    })
                    break  # one negative keyword per item is enough

        # --- positive scan (within positive lookback) ---
        if pub is None or days <= pos_days:
            for pat, kw in zip(_POS_PATTERNS, POSITIVE_KEYWORDS):
                m = pat.search(text)
                if m:
                    positive_matches.append({
                        "title": item.get("title", ""),
                        "source": item.get("source", ""),
                        "keyword": kw,
                        "published": str(pub) if pub else None,
                        "weight": weight,
                    })
                    break

    # Fuzzy dedup
    negative_matches = _dedup(negative_matches, dedup_thresh)
    positive_matches = _dedup(positive_matches, dedup_thresh)

    weighted_pos = sum(m["weight"] for m in positive_matches)

    # Decision
    if negative_matches:
        passed = False
        label = "negative"
        logger.info(f"{symbol}: Layer-3 FAIL — {len(negative_matches)} negative match(es)")
    elif all_sources_failed:
        passed = True
        label = "neutral"
        logger.debug(f"{symbol}: Layer-3 PASS (all sources failed, benefit of doubt)")
    elif weighted_pos >= min_catalyst:
        passed = True
        label = "positive"
        logger.debug(f"{symbol}: Layer-3 PASS — weighted catalyst count {weighted_pos:.2f}")
    else:
        passed = True
        label = "neutral"
        logger.debug(f"{symbol}: Layer-3 PASS (neutral, no strong negative signal)")

    return SentimentResult(
        passed=passed,
        label=label,
        positive_matches=positive_matches,
        negative_matches=negative_matches,
        weighted_count=weighted_pos,
        details={
            "symbol": symbol,
            "items_scanned": len(news_items),
            "entity_hits": len(positive_matches) + len(negative_matches),
            "all_sources_failed": all_sources_failed,
        },
    )
