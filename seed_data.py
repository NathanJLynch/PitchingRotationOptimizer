# seed_data.py
"""
First-run database initializer.

On a fresh database this is equivalent to running the server for the
first time — divisions, teams, pitchers, standings, schedule, and
recent game starts are all populated via the same sync pipeline the
server uses on every startup.

Usage:
    python seed_data.py

After this completes, just run the server normally:
    uvicorn main:app --reload

The server will keep the DB current automatically via sync on startup.
You only need to re-run this script if you want to wipe and rebuild
the database from scratch (e.g. new season, schema migration).
"""

import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///rotation_optimizer.db")


def main():
    print("=" * 52)
    print("MLB Rotation Optimizer — Initial Seed")
    print("=" * 52)
    print()
    print("This will populate the database with:")
    print("  • All 30 MLB teams and 6 divisions")
    print("  • Starting rotations for all 30 teams")
    print("  • Current standings (both leagues)")
    print("  • Upcoming schedule (45 days)")
    print("  • Recent game starts for fatigue history")
    print()

    try:
        from database.models import Base, get_engine, get_session_factory, init_db
        from database.pipeline.sync import sync_all
    except ImportError as e:
        print(f"[error] Import failed: {e}")
        print("Make sure you're running from the project root with dependencies installed.")
        sys.exit(1)

    # Wipe if requested
    if "--fresh" in sys.argv:
        confirm = input("This will DELETE all existing data. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
        import os as _os
        db_path = DATABASE_URL.replace("sqlite:///", "")
        if _os.path.exists(db_path):
            _os.remove(db_path)
            print(f"Deleted {db_path}")

    # Create tables if needed
    engine = get_engine(DATABASE_URL)
    init_db(DATABASE_URL)   # creates tables, no-op if already exist
    sf = get_session_factory(engine)

    print("Starting sync (this takes ~30-60s on first run)...")
    print()
    t0 = time.time()

    try:
        sync_all(sf)
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    print()
    print(f"{'─' * 52}")
    print(f"Done in {elapsed:.1f}s")
    print()
    print("Start the server with:")
    print("    uvicorn main:app --reload")
    print(f"{'─' * 52}")


if __name__ == "__main__":
    main()