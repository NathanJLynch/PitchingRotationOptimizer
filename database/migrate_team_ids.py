
# import os
# from database.models import get_engine, get_session_factory, get_db_session, Schedule

# DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///rotation_optimizer.db")

# # Old DB id → New DB id  (only the 5 NL Central teams that were seeded before)
# OLD_TO_NEW = {
#     1: 21,   # Brewers:   was 1,  now 21
#     2: 24,   # Reds:      was 2,  now 24
#     3: 22,   # Cubs:      was 3,  now 22
#     4: 23,   # Cardinals: was 4,  now 23
#     5: 25,   # Pirates:   was 5,  now 25
# }


# def migrate():
#     engine          = get_engine(DATABASE_URL)
#     session_factory = get_session_factory(engine)

#     with get_db_session(session_factory) as db:
#         rows = db.query(Schedule).all()
#         print(f"Found {len(rows)} schedule rows to check...")

#         updated = 0
#         for row in rows:
#             changed = False
#             if row.home_team_id in OLD_TO_NEW:
#                 row.home_team_id = OLD_TO_NEW[row.home_team_id]
#                 changed = True
#             if row.away_team_id in OLD_TO_NEW:
#                 row.away_team_id = OLD_TO_NEW[row.away_team_id]
#                 changed = True
#             if changed:
#                 updated += 1

#         db.commit()
#         print(f"Updated {updated} schedule rows.")

#     print("Migration complete. Safe to restart main.py.")


# if __name__ == "__main__":
#     migrate()