# # database/audit_pitchers.py
# """
# Diagnoses why the active-starter count is higher than expected.

# Checks for:
#   1. Duplicate mlb_player_id rows (same real pitcher inserted twice)
#   2. Pitchers with is_starter=True whose team_id no longer matches
#      any seeded team (orphaned from an old team-id remap)
#   3. Pitchers per team exceeding MAX_STARTERS_PER_TEAM (backfill
#      logic bug, or multiple seed runs without cleanup)
#   4. Null/duplicate names that suggest bad API matches

# Read-only by default — prints findings. Pass --fix to actually
# deactivate/delete the problems found.

#     python -m database.audit_pitchers
#     python -m database.audit_pitchers --fix
# """
# import os
# import sys
# from collections import defaultdict

# from database.models import get_engine, get_session_factory, get_db_session, Pitcher, Team

# DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///rotation_optimizer.db")
# EXPECTED_MAX_PER_TEAM = 6


# def audit(fix: bool = False):
#     engine          = get_engine(DATABASE_URL)
#     session_factory = get_session_factory(engine)

#     with get_db_session(session_factory) as db:
#         all_pitchers = db.query(Pitcher).all()
#         all_team_ids = {t.id for t in db.query(Team).all()}

#         print(f"Total pitcher rows: {len(all_pitchers)}")
#         print(f"Active starters:    {sum(1 for p in all_pitchers if p.is_starter and p.is_active)}")
#         print(f"Seeded teams:       {len(all_team_ids)}\n")

#         # ── 1. Duplicate mlb_player_id ──────────────────────────
#         by_mlb_id = defaultdict(list)
#         for p in all_pitchers:
#             by_mlb_id[p.mlb_player_id].append(p)

#         dupes = {k: v for k, v in by_mlb_id.items() if len(v) > 1}
#         print(f"── Duplicate mlb_player_id groups: {len(dupes)} ──")
#         dupe_ids_to_remove = []
#         for mlb_id, rows in dupes.items():
#             rows_sorted = sorted(rows, key=lambda p: p.id)
#             keep = rows_sorted[-1]   # keep the most recently inserted row
#             remove = rows_sorted[:-1]
#             print(f"  mlb_player_id={mlb_id} ({keep.name}): "
#                   f"{len(rows)} rows, keeping id={keep.id}, "
#                   f"removing ids={[r.id for r in remove]}")
#             dupe_ids_to_remove.extend(r.id for r in remove)

#         # ── 2. Orphaned team_id (points at a team that doesn't exist) ──
#         orphaned = [p for p in all_pitchers if p.team_id is not None and p.team_id not in all_team_ids]
#         print(f"\n── Orphaned team_id (team no longer exists): {len(orphaned)} ──")
#         for p in orphaned[:20]:
#             print(f"  id={p.id} {p.name}: team_id={p.team_id} (invalid)")
#         if len(orphaned) > 20:
#             print(f"  ... and {len(orphaned) - 20} more")

#         # ── 3. Teams exceeding expected starter count ───────────
#         by_team = defaultdict(list)
#         for p in all_pitchers:
#             if p.is_starter and p.is_active and p.team_id in all_team_ids:
#                 by_team[p.team_id].append(p)

#         over_cap = {tid: rows for tid, rows in by_team.items() if len(rows) > EXPECTED_MAX_PER_TEAM}
#         print(f"\n── Teams exceeding {EXPECTED_MAX_PER_TEAM} active starters: {len(over_cap)} ──")
#         for tid, rows in over_cap.items():
#             team = db.query(Team).filter_by(id=tid).first()
#             team_name = team.name if team else f"team_id={tid}"
#             print(f"  {team_name}: {len(rows)} pitchers — {', '.join(p.name for p in rows)}")

#         # ── 4. is_active pitchers with no team at all ────────────
#         no_team = [p for p in all_pitchers if p.is_starter and p.is_active and p.team_id is None]
#         print(f"\n── Active starters with no team_id: {len(no_team)} ──")
#         for p in no_team[:10]:
#             print(f"  id={p.id} {p.name}")

#         # ── Summary math ──────────────────────────────────────
#         expected_max = len(all_team_ids) * EXPECTED_MAX_PER_TEAM
#         actual = sum(1 for p in all_pitchers if p.is_starter and p.is_active)
#         print(f"\nExpected ceiling: {len(all_team_ids)} teams × {EXPECTED_MAX_PER_TEAM} = {expected_max}")
#         print(f"Actual active starters: {actual}")
#         print(f"Excess: {actual - expected_max}")

#         # ── Fix mode ────────────────────────────────────────────
#         if fix:
#             print("\n--- APPLYING FIXES ---")

#             if dupe_ids_to_remove:
#                 deleted = db.query(Pitcher).filter(Pitcher.id.in_(dupe_ids_to_remove)).delete(synchronize_session=False)
#                 print(f"Deleted {deleted} duplicate pitcher rows.")

#             if orphaned:
#                 for p in orphaned:
#                     p.is_active  = False
#                     p.is_starter = False
#                 print(f"Deactivated {len(orphaned)} orphaned-team pitchers.")

#             for tid, rows in over_cap.items():
#                 # Keep only the EXPECTED_MAX_PER_TEAM most recently inserted
#                 # (highest id = most recently seeded, generally the best data)
#                 rows_sorted = sorted(rows, key=lambda p: p.id, reverse=True)
#                 excess = rows_sorted[EXPECTED_MAX_PER_TEAM:]
#                 for p in excess:
#                     p.is_active = False
#                     p.is_starter = False
#                 if excess:
#                     print(f"  Deactivated {len(excess)} excess pitchers for team_id={tid}")

#             db.commit()
#             print("\nFixes committed. Re-run without --fix to verify.")
#         else:
#             print("\n(Read-only mode — re-run with --fix to apply corrections)")


# if __name__ == "__main__":
#     fix_mode = "--fix" in sys.argv
#     audit(fix=fix_mode)