"""Email report module for NSE Stock Screener.

Sends formatted HTML email with shortlisted stocks after the screener run.
Requires the following environment variables:
  - EMAIL_SENDER: sender email address
  - EMAIL_PASSWORD: sender email app password (use Gmail App Password)
  - EMAIL_RECIPIENTS: comma-separated list of recipient email addresses
  - SMTP_HOST: SMTP server host (default: smtp.gmail.com)
  - SMTP_PORT: SMTP server port (default: 587)
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger("email_report")


def load_latest_results(output_dir: str = "output") -> Dict[str, Any]:
    """Load the most recent JSON results file from the output directory."""
    output_path = Path(output_dir)
    json_files = sorted(output_path.glob("screener_*.json"), reverse=True)
    if not json_files:
        raise FileNotFoundError(f"No screener results found in {output_dir}")
    latest = json_files[0]
    LOGGER.info("Loading results from: %s", latest)
    return json.loads(latest.read_text(encoding="utf-8"))


def format_html_report(data: Dict[str, Any]) -> str:
    """Format screener results into an HTML email body."""
    generated_at = data.get("generated_at", datetime.now().isoformat())
    results = data.get("results", [])
    count = data.get("count", len(results))

    if not results:
        return _format_empty_report(generated_at)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
  .container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .header {{ background: linear-gradient(135deg, #1a237e, #0d47a1); color: white; padding: 24px; text-align: center; }}
  .header h1 {{ margin: 0; font-size: 22px; }}
  .header p {{ margin: 8px 0 0; opacity: 0.85; font-size: 13px; }}
  .summary {{ padding: 16px 24px; background: #e8f5e9; border-bottom: 1px solid #c8e6c9; }}
  .summary p {{ margin: 4px 0; color: #2e7d32; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f5f5f5; padding: 10px 12px; text-align: left; font-weight: 600; color: #333; border-bottom: 2px solid #ddd; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #eee; color: #555; }}
  tr:hover td {{ background: #f9f9f9; }}
  .stock-card {{ margin: 16px 24px; padding: 16px; border: 1px solid #e0e0e0; border-radius: 6px; background: #fafafa; }}
  .stock-card h3 {{ margin: 0 0 12px; color: #1a237e; }}
  .stock-card .row {{ display: flex; justify-content: space-between; padding: 4px 0; }}
  .stock-card .label {{ color: #666; font-size: 12px; }}
  .stock-card .value {{ font-weight: 600; color: #333; font-size: 13px; }}
  .targets {{ color: #2e7d32; }}
  .stoploss {{ color: #c62828; }}
  .footer {{ padding: 16px 24px; background: #fff3e0; border-top: 1px solid #ffe0b2; font-size: 11px; color: #e65100; text-align: center; }}
  .rank {{ display: inline-block; width: 24px; height: 24px; line-height: 24px; text-align: center; border-radius: 50%; background: #1a237e; color: white; font-weight: bold; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>📈 NSE Stock Screener — Top {count} Under ₹300</h1>
    <p>Generated: {generated_at} IST | Monday–Friday Pre-Market Report</p>
  </div>
  <div class="summary">
    <p>🎯 {count} stock(s) shortlisted based on technical + fundamental analysis</p>
  </div>

  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Symbol</th>
        <th>Price</th>
        <th>Sector</th>
        <th>Confidence</th>
        <th>Entry</th>
        <th>Target 1</th>
        <th>Stop Loss</th>
        <th>Return</th>
      </tr>
    </thead>
    <tbody>
"""
    for idx, stock in enumerate(results, 1):
        symbol = stock.get("symbol", "N/A")
        price = stock.get("current_price", 0)
        sector = stock.get("sector", "N/A")
        confidence = stock.get("confidence_score", 0)
        entry = stock.get("entry", 0)
        target1 = stock.get("target1", 0)
        stop_loss = stock.get("stop_loss", 0)
        return_min = stock.get("return_min", 0)
        return_max = stock.get("return_max", 0)

        html += f"""      <tr>
        <td><span class="rank">{idx}</span></td>
        <td><strong>{symbol}</strong></td>
        <td>₹{price:.2f}</td>
        <td>{sector}</td>
        <td><strong>{confidence:.1f}</strong>/100</td>
        <td>₹{entry:.2f}</td>
        <td class="targets">₹{target1:.2f}</td>
        <td class="stoploss">₹{stop_loss:.2f}</td>
        <td>{return_min:.1f}%–{return_max:.1f}%</td>
      </tr>
"""

    html += """    </tbody>
  </table>
"""

    # Detail cards for each stock
    for idx, stock in enumerate(results, 1):
        symbol = stock.get("symbol", "N/A")
        company = stock.get("company", "N/A")
        price = stock.get("current_price", 0)
        sector = stock.get("sector", "N/A")
        entry = stock.get("entry", 0)
        stop_loss = stock.get("stop_loss", 0)
        stop_loss_pct = stock.get("stop_loss_pct", 0)
        target1 = stock.get("target1", 0)
        target2 = stock.get("target2", 0)
        target3 = stock.get("target3", 0)
        rr1 = stock.get("rr1", 0)
        rr2 = stock.get("rr2", 0)
        rr3 = stock.get("rr3", 0)
        technical_score = stock.get("technical_score", 0)
        fundamental_score = stock.get("fundamental_score", 0)
        confidence = stock.get("confidence_score", 0)
        rsi = stock.get("rsi", 0)
        adx = stock.get("adx", 0)
        position_qty = stock.get("position_qty", 0)
        position_value = stock.get("position_value", 0)
        validity = stock.get("validity_days", 2)
        triggers = stock.get("trigger_events", [])
        risk_flags = stock.get("risk_flags", [])
        patterns = stock.get("bullish_patterns", [])

        triggers_str = ", ".join(triggers[:4]) if triggers else "Momentum setup"
        risk_str = ", ".join(risk_flags) if risk_flags else "None"
        patterns_str = ", ".join(patterns) if patterns else "None"

        html += f"""
  <div class="stock-card">
    <h3><span class="rank">{idx}</span> {symbol} — {company}</h3>
    <div class="row"><span class="label">Price / Sector</span><span class="value">₹{price:.2f} | {sector}</span></div>
    <div class="row"><span class="label">Entry Level</span><span class="value">₹{entry:.2f}</span></div>
    <div class="row"><span class="label">🎯 Target 1 (R:R {rr1:.1f}:1)</span><span class="value targets">₹{target1:.2f}</span></div>
    <div class="row"><span class="label">🎯 Target 2 (R:R {rr2:.1f}:1)</span><span class="value targets">₹{target2:.2f}</span></div>
    <div class="row"><span class="label">🎯 Target 3 (R:R {rr3:.1f}:1)</span><span class="value targets">₹{target3:.2f}</span></div>
    <div class="row"><span class="label">🛑 Stop Loss</span><span class="value stoploss">₹{stop_loss:.2f} ({stop_loss_pct:.2f}%)</span></div>
    <div class="row"><span class="label">Technical Score</span><span class="value">{technical_score:.1f}/100 | RSI {rsi:.1f} | ADX {adx:.1f}</span></div>
    <div class="row"><span class="label">Fundamental Score</span><span class="value">{fundamental_score:.1f}/100</span></div>
    <div class="row"><span class="label">Overall Confidence</span><span class="value">{confidence:.1f}/100</span></div>
    <div class="row"><span class="label">Trigger Events</span><span class="value">{triggers_str}</span></div>
    <div class="row"><span class="label">Bullish Patterns</span><span class="value">{patterns_str}</span></div>
    <div class="row"><span class="label">Risk Flags</span><span class="value">{risk_str}</span></div>
    <div class="row"><span class="label">Position Sizing</span><span class="value">{position_qty} shares (~₹{position_value:,.0f})</span></div>
    <div class="row"><span class="label">Validity</span><span class="value">Next {validity} trading sessions</span></div>
  </div>
"""

    html += """
  <div class="footer">
    ⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Uses Yahoo Finance data which may be delayed or incomplete.
    Always verify liquidity, corporate actions, and risk before trading. Past performance does not guarantee future results.
  </div>
</div>
</body>
</html>"""
    return html


def _format_empty_report(generated_at: str) -> str:
    """Format an email for when no stocks are shortlisted."""
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: Arial, sans-serif; padding: 20px;">
  <h2 style="color: #c62828;">📊 NSE Stock Screener — No Stocks Shortlisted</h2>
  <p>Generated: {generated_at}</p>
  <p>No stocks met all shortlist criteria in today's scan. This can happen when:</p>
  <ul>
    <li>Market conditions are weak across the board</li>
    <li>Price, volume, or trend filters are restrictive</li>
    <li>Fundamental data is temporarily unavailable</li>
  </ul>
  <p>The screener will run again on the next trading day.</p>
  <hr>
  <p style="font-size: 11px; color: #999;">This is an automated report from NSE Under ₹300 Stock Screener.</p>
</body>
</html>"""


def send_email(html_body: str, subject: str | None = None) -> bool:
    """Send the HTML report via SMTP email."""
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    recipients_str = os.environ.get("EMAIL_RECIPIENTS", "")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    if not sender or not password or not recipients_str:
        LOGGER.error(
            "Email configuration incomplete. Set EMAIL_SENDER, EMAIL_PASSWORD, and EMAIL_RECIPIENTS environment variables."
        )
        return False

    recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
    if not recipients:
        LOGGER.error("No valid recipients found in EMAIL_RECIPIENTS.")
        return False

    if subject is None:
        today = datetime.now().strftime("%d %b %Y")
        subject = f"📈 NSE Stock Screener Report — {today}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    # Plain text fallback
    plain_text = (
        "Your NSE Stock Screener report is ready. "
        "Please view this email in an HTML-compatible email client for the formatted report."
    )
    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        LOGGER.info("Connecting to SMTP server %s:%d", smtp_host, smtp_port)
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        LOGGER.info("Email sent successfully to: %s", ", ".join(recipients))
        return True
    except smtplib.SMTPException as exc:
        LOGGER.error("Failed to send email: %s", exc)
        return False


def main() -> int:
    """Load latest results and send email report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = os.environ.get("OUTPUT_DIR", "output")

    try:
        data = load_latest_results(output_dir)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return 1

    html_body = format_html_report(data)
    success = send_email(html_body)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
