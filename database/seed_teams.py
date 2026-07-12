# database/seed_teams.py
"""
Seeds all 30 MLB teams and 6 divisions into the database.
Safe to re-run — uses upsert logic so existing rows aren't duplicated.

Run once:
    python -m database.seed_teams
"""
import os
from database.models import get_engine, get_session_factory, get_db_session, Division, Team

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///rotation_optimizer.db")

DIVISIONS = [
    {"id": 1, "name": "AL_EAST",    "league": "AL", "display_name": "AL East"},
    {"id": 2, "name": "AL_CENTRAL", "league": "AL", "display_name": "AL Central"},
    {"id": 3, "name": "AL_WEST",    "league": "AL", "display_name": "AL West"},
    {"id": 4, "name": "NL_EAST",    "league": "NL", "display_name": "NL East"},
    {"id": 5, "name": "NL_CENTRAL", "league": "NL", "display_name": "NL Central"},
    {"id": 6, "name": "NL_WEST",    "league": "NL", "display_name": "NL West"},
]

# (db_id, mlb_team_id, name, abbreviation, division_db_id)
TEAMS = [
    # AL East
    (1,  147, "New York Yankees",      "NYY", 1),
    (2,  111, "Boston Red Sox",        "BOS", 1),
    (3,  141, "Toronto Blue Jays",     "TOR", 1),
    (4,  110, "Baltimore Orioles",     "BAL", 1),
    (5,  139, "Tampa Bay Rays",        "TB",  1),
    # AL Central
    (6,  142, "Minnesota Twins",       "MIN", 2),
    (7,  116, "Detroit Tigers",        "DET", 2),
    (8,  118, "Kansas City Royals",    "KC",  2),
    (9,  114, "Cleveland Guardians",   "CLE", 2),
    (10, 145, "Chicago White Sox",     "CWS", 2),
    # AL West
    (11, 117, "Houston Astros",        "HOU", 3),
    (12, 140, "Texas Rangers",         "TEX", 3),
    (13, 108, "Los Angeles Angels",    "LAA", 3),
    (14, 136, "Seattle Mariners",      "SEA", 3),
    (15, 133, "Oakland Athletics",     "OAK", 3),
    # NL East
    (16, 121, "New York Mets",         "NYM", 4),
    (17, 143, "Philadelphia Phillies", "PHI", 4),
    (18, 144, "Atlanta Braves",        "ATL", 4),
    (19, 120, "Washington Nationals",  "WSH", 4),
    (20, 146, "Miami Marlins",         "MIA", 4),
    # NL Central
    (21, 158, "Milwaukee Brewers",     "MIL", 5),
    (22, 112, "Chicago Cubs",          "CHC", 5),
    (23, 138, "St. Louis Cardinals",   "STL", 5),
    (24, 113, "Cincinnati Reds",       "CIN", 5),
    (25, 134, "Pittsburgh Pirates",    "PIT", 5),
    # NL West
    (26, 119, "Los Angeles Dodgers",   "LAD", 6),
    (27, 137, "San Francisco Giants",  "SF",  6),
    (28, 135, "San Diego Padres",      "SD",  6),
    (29, 109, "Arizona Diamondbacks",  "ARI", 6),
    (30, 115, "Colorado Rockies",      "COL", 6),
]

# Updated map matching new DB IDs above — paste this into sync.py
MLB_TO_DB_TEAM_NEW = {mlb_id: db_id for db_id, mlb_id, *_ in TEAMS}


def seed():
    engine          = get_engine(DATABASE_URL)
    session_factory = get_session_factory(engine)

    with get_db_session(session_factory) as db:
        # ── Divisions ──────────────────────────────────────────
        print("Seeding divisions...")
        for d in DIVISIONS:
            existing = db.query(Division).filter_by(id=d["id"]).first()
            if existing:
                existing.name         = d["name"]
                existing.league       = d["league"]
                existing.display_name = d["display_name"]
            else:
                db.add(Division(**d))
        db.flush()

        # ── Teams ──────────────────────────────────────────────
        print("Seeding teams...")
        for db_id, mlb_id, name, abbrev, div_id in TEAMS:
            existing = db.query(Team).filter_by(mlb_team_id=mlb_id).first()
            if existing:
                # Update fields but preserve any stats already populated
                existing.id           = db_id
                existing.name         = name
                existing.abbreviation = abbrev
                existing.division_id  = div_id
                print(f"  ↺ Updated  {name} (db_id={db_id})")
            else:
                db.add(Team(
                    id            = db_id,
                    mlb_team_id   = mlb_id,
                    name          = name,
                    abbreviation  = abbrev,
                    division_id   = div_id,
                ))
                print(f"  ✓ Inserted {name} (db_id={db_id})")

        db.commit()

    print(f"\nDone — {len(TEAMS)} teams, {len(DIVISIONS)} divisions seeded.")
    print("\nUpdate MLB_TO_DB_TEAM in database/pipeline/sync.py to:")
    print("{")
    for db_id, mlb_id, name, abbrev, div_id in TEAMS:
        print(f"    {mlb_id}: {db_id},   # {abbrev} — {name}")
    print("}")


if __name__ == "__main__":
    seed()