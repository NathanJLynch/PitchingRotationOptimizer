# database/seed_pitchers.py
"""
Seeds active starting pitchers for all 30 MLB teams.

Identifies real rotation starters (not relievers) using actual
season games-started stats rather than roster order:

  1. Pull season pitching stats for the whole league
  2. Compute innings-per-start (IP / GS) for everyone with GS > 0
  3. Set the eligibility bar at the STARTER_PERCENTILE-th percentile
     of IP/start among pitchers with starts (a floor for "this guy
     starts games and goes deep enough to matter"), with a 4.0
     IP/start hard minimum so a small-sample fluke can't lower it.
     STARTER_PERCENTILE is tunable below — lower it to qualify more
     pitchers per team, raise it to be stricter.
  4. Cross-reference team rosters against that qualifying set
  5. If a team has fewer than MIN_BACKFILL_STARTERS qualifiers,
     backfill with next-best roster pitchers — but ONLY if they
     clear MIN_BACKFILL_IP_PER_START (3.5 IP/start). A team can and
     will end up with fewer than 5 "starters" if that's the truth;
     we never pad a rotation with a 1-2 inning reliever just to hit
     a target headcount.

Run once after seed_teams.py. Safe to re-run — upserts by mlb_player_id.

    python -m database.seed_pitchers
"""
import os
import requests
import logging
import numpy as np
from datetime import date

from database.models import get_engine, get_session_factory, get_db_session, Team, Pitcher, GameStart

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///rotation_optimizer.db")

# Cap per team in case more than 5-6 pitchers qualify (injuries, call-ups, etc.)
MAX_STARTERS_PER_TEAM = 6

# Which percentile of IP/start (among pitchers with real sample size)
# sets the eligibility bar. Lower = more pitchers qualify as starters.
# 25 was too strict in practice (left several teams with only 2-3
# qualifying starters) — 10 is a looser floor that still excludes
# true long-relievers/openers.
STARTER_PERCENTILE = 10

# Hard floor — even if the percentile above computes lower than this
# (e.g. early season, small samples), never call someone a starter
# below 4 IP/start on average.
MIN_IP_PER_START = 4.0

# Minimum starts before a pitcher's IP/start average is trusted at all —
# below this, one short outing skews the average wildly
MIN_GAMES_STARTED = 3

# If a team has fewer than this many pitchers who clear the IP/start
# bar, backfill with their next-best roster pitchers (by IP/start,
# even if below the qualifying threshold) up to this count. Keeps
# the optimizer usable for thin rotations without padding with
# true relievers. Backfilled pitchers are flagged in the log output.
MIN_BACKFILL_STARTERS = 5

# Hard floor for backfill candidates. A pitcher averaging under this
# many IP/start is a reliever/opener, full stop — never let them in
# as a "starter" even if a team is short on real qualifiers. Better
# to have 2 real starters than 5 padded with 1-inning relievers.
MIN_BACKFILL_IP_PER_START = 3.5

# ── Ground-truth cross-check ───────────────────────────────────
# The MLB stats API's season gamesStarted/inningsPitched line can be
# stale, cross-season, or just wrong for recent call-ups — that's how
# closers like Bednar/Hader/Erceg were getting flagged is_starter=True
# despite never starting a game. Once we have real boxscore data
# synced (GameStart, populated by sync.py's _sync_game_starts), it's
# ground truth and should override the API-only signal: if a pitcher
# has GameStart history AND it shows them as a reliever (low IP/outing),
# don't mark them a starter even if the season-stats API thinks so.
MIN_REAL_GAMESTART_IP_TO_TRUST = 3.0


def seed():
    engine          = get_engine(DATABASE_URL)
    session_factory = get_session_factory(engine)

    logger.info("Fetching league-wide pitching stats to determine starter threshold...")
    league_stats = _fetch_league_pitching_stats(date.today().year)

    qualifying_ids, threshold, all_ip_per_start = _compute_qualifying_starters(league_stats)
    logger.info(
        f"Starter threshold: {threshold:.2f} IP/start "
        f"({len(qualifying_ids)} qualifying pitchers league-wide)"
    )

    with get_db_session(session_factory) as db:
        teams = db.query(Team).all()
        logger.info(f"Seeding pitchers for {len(teams)} teams...")

        total_inserted = 0
        total_updated  = 0

        for team in teams:
            try:
                roster_pitchers = _fetch_team_pitchers(team.mlb_team_id)
            except Exception as e:
                logger.warning(f"  - {team.name}: roster fetch failed ({e})")
                continue

            if not roster_pitchers:
                logger.warning(f"  - {team.name}: no pitchers found on roster")
                continue

            # Filter roster to only pitchers who clear the starter bar
            qualified = [
                p for p in roster_pitchers
                if p["mlb_player_id"] in qualifying_ids
            ]

            # Sort by IP/start descending so the best starters come first
            qualified.sort(
                key=lambda p: qualifying_ids.get(p["mlb_player_id"], 0),
                reverse=True,
            )

            n_real_qualifiers = len(qualified)
            backfilled_names = []

            # ── Backfill thin rotations ────────────────────────────
            # If fewer than MIN_BACKFILL_STARTERS cleared the bar,
            # pull in next-best roster pitchers by IP/start — but
            # ONLY if they clear MIN_BACKFILL_IP_PER_START. A pitcher
            # averaging 1-2 IP/outing is a reliever no matter how it's
            # framed; we'd rather ship a thin rotation than a wrong one.
            if n_real_qualifiers < MIN_BACKFILL_STARTERS:
                qualified_ids_set = {p["mlb_player_id"] for p in qualified}
                candidates = [
                    p for p in roster_pitchers
                    if p["mlb_player_id"] not in qualified_ids_set
                    and all_ip_per_start.get(p["mlb_player_id"], 0) >= MIN_BACKFILL_IP_PER_START
                ]
                candidates.sort(
                    key=lambda p: all_ip_per_start.get(p["mlb_player_id"], 0),
                    reverse=True,
                )
                needed = MIN_BACKFILL_STARTERS - n_real_qualifiers
                backfill = candidates[:needed]
                backfilled_names = [p["name"] for p in backfill]
                qualified.extend(backfill)

            qualified = qualified[:MAX_STARTERS_PER_TEAM]

            if not qualified:
                logger.warning(
                    f"  - {team.name}: 0 pitchers found at all (real or backfill) "
                    f"— roster fetch may have returned no pitchers"
                )
                continue

            if len(qualified) < MIN_BACKFILL_STARTERS and not backfilled_names:
                logger.info(
                    f"  ! {team.name}: only {len(qualified)} pitchers clear "
                    f"{MIN_BACKFILL_IP_PER_START} IP/start — shipping a thin "
                    f"rotation rather than padding with relievers"
                )

            inserted_for_team = 0
            demoted_by_ground_truth = []

            for p in qualified:
                existing = db.query(Pitcher).filter_by(
                    mlb_player_id=p["mlb_player_id"]
                ).first()

                # ── Ground-truth check ─────────────────────────────
                # If we have real synced GameStart rows for this pitcher,
                # trust them over the MLB stats API's season line. A
                # pitcher with real starts averaging under the IP floor
                # is a reliever — don't flag them, no matter what the
                # API's gamesStarted/inningsPitched said.
                is_starter_flag = True
                if existing:
                    real_starts = db.query(GameStart).filter_by(pitcher_id=existing.id).all()
                    if real_starts:
                        avg_real_ip = sum(s.innings_pitched for s in real_starts) / len(real_starts)
                        if avg_real_ip < MIN_REAL_GAMESTART_IP_TO_TRUST:
                            is_starter_flag = False
                            demoted_by_ground_truth.append(f"{p['name']} ({avg_real_ip:.2f} IP avg)")

                if existing:
                    existing.name       = p["name"]
                    existing.team_id    = team.id
                    existing.throws     = p["throws"]
                    existing.is_starter = is_starter_flag
                    existing.is_active  = True
                    total_updated += 1
                else:
                    db.add(Pitcher(
                        mlb_player_id = p["mlb_player_id"],
                        name          = p["name"],
                        team_id       = team.id,
                        throws        = p["throws"],
                        is_starter    = True,
                        is_active     = True,
                    ))
                    total_inserted += 1
                inserted_for_team += 1

            db.flush()
            if demoted_by_ground_truth:
                logger.info(
                    f"  ~ {team.name}: demoted by real GameStart data: "
                    f"{', '.join(demoted_by_ground_truth)}"
                )
            if backfilled_names:
                logger.info(
                    f"  + {team.name}: {inserted_for_team} starters "
                    f"({n_real_qualifiers} qualified, backfilled: {', '.join(backfilled_names)})"
                )
            else:
                logger.info(f"  + {team.name}: {inserted_for_team} qualified starters")

        db.commit()

    logger.info(f"\nDone — {total_inserted} new pitchers, {total_updated} updated.")
    logger.info("Run `python main.py` to populate their Statcast stats via the sync.")


def _fetch_league_pitching_stats(season: int) -> list[dict]:
    """
    Pull season pitching stats for the whole league in one call.
    Returns list of dicts with mlb_player_id, games_started, innings_pitched.
    """
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={season}&playerPool=ALL&limit=2000"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for split in data.get("stats", [{}])[0].get("splits", []):
        stat   = split.get("stat", {})
        person = split.get("player", {})

        gs = int(stat.get("gamesStarted", 0) or 0)
        if gs == 0:
            continue   # not relevant to starter detection

        # inningsPitched comes as a string like "142.1" (the .1 = 1/3 inning)
        ip_raw = stat.get("inningsPitched", "0.0")
        ip = _parse_innings(ip_raw)

        rows.append({
            "mlb_player_id": person.get("id"),
            "games_started": gs,
            "innings_pitched": ip,
        })

    return rows


def _parse_innings(ip_str: str) -> float:
    """MLB innings notation: '142.1' means 142 + 1/3 innings, '142.2' means 142 + 2/3."""
    try:
        whole, _, frac = str(ip_str).partition(".")
        whole = float(whole or 0)
        frac_map = {"0": 0.0, "1": 1 / 3, "2": 2 / 3}
        return whole + frac_map.get(frac, 0.0)
    except Exception:
        return 0.0


def _compute_qualifying_starters(league_stats: list[dict]) -> tuple[dict, float, dict]:
    """
    Returns (qualifying_ids, threshold, all_ip_per_start):
      qualifying_ids    — mlb_player_id -> IP/start, for pitchers who clear the bar
      threshold         — the IP/start cutoff that was applied
      all_ip_per_start  — mlb_player_id -> IP/start for EVERY pitcher with a start,
                           qualifying or not (used to rank backfill candidates)

    threshold = max(STARTER_PERCENTILE-th percentile of IP/start, MIN_IP_PER_START)
    Only pitchers with >= MIN_GAMES_STARTED are included in the
    percentile calculation, to avoid one spot-start skewing things.
    """
    eligible = [
        row for row in league_stats
        if row["games_started"] >= MIN_GAMES_STARTED
    ]

    if not eligible:
        # Early season fallback — not enough data yet, use hard minimum only
        ids = {
            row["mlb_player_id"]: row["innings_pitched"] / max(row["games_started"], 1)
            for row in league_stats
            if row["games_started"] > 0
        }
        return ids, MIN_IP_PER_START, ids

    ip_per_start = np.array([
        row["innings_pitched"] / row["games_started"] for row in eligible
    ])

    p25 = float(np.percentile(ip_per_start, STARTER_PERCENTILE))
    threshold = max(p25, MIN_IP_PER_START)

    qualifying_ids = {
        row["mlb_player_id"]: row["innings_pitched"] / row["games_started"]
        for row in eligible
        if (row["innings_pitched"] / row["games_started"]) >= threshold
    }

    # Also compute IP/start for every pitcher with at least 1 start,
    # regardless of qualifying — used for backfilling thin rotations
    all_ip_per_start = {
        row["mlb_player_id"]: row["innings_pitched"] / max(row["games_started"], 1)
        for row in league_stats
    }

    return qualifying_ids, threshold, all_ip_per_start


def _fetch_team_pitchers(mlb_team_id: int) -> list[dict]:
    """
    Fetch the active roster for a team, filtered to pitchers,
    with handedness filled in via the people endpoint.
    """
    url = f"https://statsapi.mlb.com/api/v1/teams/{mlb_team_id}/roster?rosterType=active"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()

    pitchers = []
    for entry in data.get("roster", []):
        position = entry.get("position", {})
        if position.get("type") != "Pitcher":
            continue

        person = entry.get("person", {})
        pitchers.append({
            "mlb_player_id": person.get("id"),
            "name":          person.get("fullName"),
            "throws":        None,
        })

    if pitchers:
        ids = ",".join(str(p["mlb_player_id"]) for p in pitchers)
        people_url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids}"
        try:
            pr = requests.get(people_url, timeout=15)
            pr.raise_for_status()
            people = {p["id"]: p for p in pr.json().get("people", [])}
            for p in pitchers:
                person = people.get(p["mlb_player_id"], {})
                throws = person.get("pitchHand", {}).get("code", "R")
                p["throws"] = throws if throws in ("R", "L") else "R"
        except Exception:
            for p in pitchers:
                p["throws"] = "R"

    return pitchers


if __name__ == "__main__":
    seed()