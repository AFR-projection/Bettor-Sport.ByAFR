"""CLI one-shot agent cycle for AI Bettor.

Runs one full automated cycle:
    The Odds API (multi-key) -> save -> analyze -> simulate (repeated batches)
    -> risk -> bettor brain -> save -> Telegram (high-score picks only)

Usage:
    python scripts/run_agent.py [--no-early-morning] [--loop N]

Can be scheduled with Windows Task Scheduler / cron to run automatically.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ai-bettor.cli")

sys.path.insert(0, ".")

from backend.database.session import init_db  # noqa: E402
from backend.integrations.odds_router import get_odds_router  # noqa: E402
from backend.services.pipeline import get_pipeline  # noqa: E402


def run_once(early_morning_only: bool) -> int:
    init_db()
    if not get_odds_router().has_keys:
        logger.warning("Tidak ada The Odds API key. Isi lewat UI Settings atau .env (THE_ODDS_API_KEY).")
        return 1

    logger.info("Mulai siklus agent (early-morning only: %s)...", early_morning_only)
    pipeline = get_pipeline()
    t0 = time.time()
    summary = pipeline.run_cycle(early_morning_only=early_morning_only)
    dt = time.time() - t0
    logger.info(
        "Selesai dalam %.1fs: %s match discan, %s kandidat BET, %s terkirim ke Telegram",
        dt,
        summary.get("matches_scanned", 0),
        summary.get("bet_candidates", 0),
        summary.get("telegram_sent", 0),
    )
    return 0 if summary.get("status") == "completed" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Bettor one-shot agent cycle")
    parser.add_argument("--no-early-morning", action="store_true",
                        help="Nonaktifkan filter dini hari (semua jam kickoff)")
    parser.add_argument("--loop", type=int, default=0,
                        help="Jalankan N siklus berurutan (0 = sekali saja)")
    args = parser.parse_args()

    early_morning_only = not args.no_early_morning
    loops = max(0, args.loop)
    attempts = loops if loops > 0 else 1
    for i in range(attempts):
        rc = run_once(early_morning_only)
        if loops > 0 and i < attempts - 1:
            logger.info("Menunggu interval sebelum siklus berikutnya...")
            time.sleep(5)
    return rc


if __name__ == "__main__":
    sys.exit(main())