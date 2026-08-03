# database/pipeline/sync.py
from datetime import date, timedelta
from typing import Optional
import asyncio
import logging

import httpx
import numpy as np

from database.models import (
    Team, Pitcher, Schedule, StandingsSnapshot,
    get_db_session
)
from database.pipeline.statcast import (
    fetch_pitcher_statcast, fetch_team_statcast
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────
# STATIC TEAM + DIVISION METADATA
# ─────────────────────────────────────────────────────────────

ALL_MLB_TEAMS: dict[int, tuple[str, str]] = {
    147: ("NYY", "AL_EAST"),   111: ("BOS", "AL_EAST"),
    141: ("TOR", "AL_EAST"),   110: ("BAL", "AL_EAST"),
    139: ("TB",  "AL_EAST"),   142: ("MIN", "AL_CENTRAL"),
    116: ("DET", "AL_CENTRAL"),118: ("KC",  "AL_CENTRAL"),
    114: ("CLE", "AL_CENTRAL"),145: ("CWS", "AL_CENTRAL"),
    117: ("HOU", "AL_WEST"),   140: ("TEX", "AL_WEST"),
    108: ("LAA", "AL_WEST"),   136: ("SEA", "AL_WEST"),
    133: ("OAK", "AL_WEST"),   121: ("NYM", "NL_EAST"),
    143: ("PHI", "NL_EAST"),   144: ("ATL", "NL_EAST"),
    120: ("WSH", "NL_EAST"),   146: ("MIA", "NL_EAST"),
    158: ("MIL", "NL_CENTRAL"),112: ("CHC", "NL_CENTRAL"),
    138: ("STL", "NL_CENTRAL"),113: ("CIN", "NL_CENTRAL"),
    134: ("PIT", "NL_CENTRAL"),119: ("LAD", "NL_WEST"),
    137: ("SF",  "NL_WEST"),   135: ("SD",  "NL_WEST"),
    109: ("ARI", "NL_WEST"),   115: ("COL", "NL_WEST"),
}

ALL_DIVISIONS = [
    ("AL_EAST",    "AL", "AL East"),
    ("AL_CENTRAL", "AL", "AL Central"),
    ("AL_WEST",    "AL", "AL West"),
    ("NL_EAST",    "NL", "NL East"),
    ("NL_CENTRAL", "NL", "NL Central"),
    ("NL_WEST",    "NL", "NL West"),
]

MLB_DIVISION_ID_TO_NAME: dict[int, str] = {
    200: "AL_WEST", 201: "AL_EAST", 202: "AL_CENTRAL",
    203: "NL_WEST", 204: "NL_EAST", 205: "NL_CENTRAL",
}

# Runtime maps — cleared and rebuilt from DB at start of every sync
# so auto-increment IDs are always correct.
MLB_TO_DB_TEAM: dict[int, int] = {}
DB_TO_MLB_TEAM: dict[int, int] = {}

SCHEDULE_HORIZON_DAYS     = 45
GAME_STARTS_LOOKBACK_DAYS = 30
MLB_API_CONCURRENCY       = 12
STATCAST_CONCURRENCY      = 5
HTTP_TIMEOUT              = 15.0

# ─────────────────────────────────────────────────────────────
# STARTER ELIGIBILITY CRITERIA
#
# A pitcher qualifies as a starter if they meet BOTH:
#   • At least STARTER_MIN_GS games started this season
#   • At least STARTER_MIN_IP_PER_APP average innings per appearance
#
# Using total appearances (gamesPitched) rather than games started
# as the IP denominator prevents two-way pitchers and swingmen
# with many relief outings from inflating their IP/start ratio
# (e.g. 3 starts + 30 relief appearances = total IP / 3 looks
# like 10+ IP per start, which is obviously wrong).
#
# This filters out:
#   - Spot starters with only 1 appearance (min GS check)
#   - Openers / bulk relievers who start but throw 2 innings
#     before handing off (IP/appearance ratio check)
# ─────────────────────────────────────────────────────────────

STARTER_MIN_GS            = 2     # must have at least 2 starts
STARTER_MIN_IP_PER_APP    = 3.1   # must average at least 3.1 IP per appearance


LEAGUE_AVG_PITCHER = {
    "fb_velo": 93.5, "spin_rate": 2200.0, "extension": 6.5,
    "whiff_pct": 0.115, "zone_pct": 0.47, "chase_pct": 0.30,
    "fastball_usage": 0.55, "breaking_usage": 0.28, "offspeed_usage": 0.17,
    "ops_allowed": 0.720, "woba_allowed": 0.310,
    "xfip": 4.10, "fip": 4.05, "hard_hit_pct": 0.37, "xwoba": 0.315,
    "k_pct": 0.220, "bb_pct": 0.075,
}

LEAGUE_AVG_TEAM = {
    "ops": 0.720, "woba": 0.315, "iso": 0.155, "babip": 0.295,
    "k_pct": 0.222, "bb_pct": 0.082, "rhb_pct": 0.58, "lhb_pct": 0.42,
    "ops_vs_fastball": 0.760, "ops_vs_breaking": 0.670, "ops_vs_offspeed": 0.700,
    "whiff_rate_60d": 0.240, "chase_pct": 0.290, "hard_hit_pct": 0.370,
    "xwoba": 0.315, "ops_60d": 0.720, "woba_60d": 0.315,
}


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINTS
# ─────────────────────────────────────────────────────────────

async def sync_all_async(session_factory, our_team_db_id: Optional[int] = None):
    """
    Async-native entry point. Await this directly from FastAPI startup:

        @app.on_event("startup")
        async def startup():
            await sync_all_async(session_factory)

    Phases:
      0. Init    — create divisions + teams if DB is empty (no-op otherwise)
      1. DB read — rebuild runtime ID maps, load pitcher/team rows
      2. Fetch   — all HTTP calls run concurrently
      3. Write   — isolated DB sessions per section
    """
    # Phase 0: first-run bootstrap — no-op if rows already exist
    try:
        with get_db_session(session_factory) as db:
            needs_init = _needs_init(db)

        if needs_init:
            team_info = await _fetch_all_team_info_for_init()
            with get_db_session(session_factory) as db:
                _write_divisions_and_teams(db, team_info)
    except Exception as e:
        logger.error(f"Init phase failed: {e}")

    # Phase 1: rebuild runtime maps + load state for fetch planning
    try:
        with get_db_session(session_factory) as db:
            all_teams = db.query(Team).all()
            MLB_TO_DB_TEAM.clear()
            DB_TO_MLB_TEAM.clear()
            for t in all_teams:
                MLB_TO_DB_TEAM[t.mlb_team_id] = t.id
                DB_TO_MLB_TEAM[t.id] = t.mlb_team_id

            pitcher_rows = [
                (p.id, p.mlb_player_id, p.name)
                for p in db.query(Pitcher).filter_by(is_starter=True, is_active=True).all()
            ]
            team_rows = [
                (t.id, t.mlb_team_id, t.name)
                for t in all_teams
            ]
            known_mlb_player_ids = {
                p.mlb_player_id for p in db.query(Pitcher).all()
            }
    except Exception as e:
        logger.error(f"Pre-fetch DB read failed: {e}")
        pitcher_rows, team_rows, known_mlb_player_ids = [], [], set()

    # Phase 2: fetch
    try:
        fetched = await _fetch_everything(
            our_team_db_id, pitcher_rows, team_rows, known_mlb_player_ids
        )
    except Exception as e:
        logger.error(f"Async fetch phase failed entirely: {e}")
        fetched = {
            "standings": [], "schedule": [], "game_starts": [],
            "pitcher_stats": [], "team_stats": [], "rosters": [],
        }

    # Phase 3: write — isolated sessions so one failure can't poison others
    for label, fn, key in [
        ("Roster reconcile", _reconcile_rosters, "rosters"),
        ("Standings",        _write_standings,   "standings"),
        ("Schedule",         _write_schedule,    "schedule"),
        ("Game starts",      _write_game_starts, "game_starts"),
        ("Pitcher stats",    _write_pitcher_stats, "pitcher_stats"),
        ("Team stats",       _write_team_stats,  "team_stats"),
    ]:
        try:
            with get_db_session(session_factory) as db:
                fn(db, fetched.get(key))
        except Exception as e:
            logger.error(f"{label} sync failed: {e}")

    logger.info("Sync complete.")


def sync_all(session_factory, our_team_db_id: Optional[int] = None):
    """
    Sync wrapper for non-async callers (scripts, cron jobs).
    Prefer `await sync_all_async(...)` if already in async context.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(sync_all_async(session_factory, our_team_db_id))
        return

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(asyncio.run, sync_all_async(session_factory, our_team_db_id)).result()


# ─────────────────────────────────────────────────────────────
# FIRST-RUN INIT — divisions + teams
#
# HTTP fetch and DB write are separated so we never hold a DB
# session open during network calls.
# ─────────────────────────────────────────────────────────────

def _needs_init(db) -> bool:
    from database.models import Division
    return db.query(Division).count() == 0 or db.query(Team).count() == 0


async def _fetch_all_team_info_for_init() -> dict[int, dict]:
    """Fetch team name/abbreviation for all 30 teams. No DB session involved."""
    logger.info("First run detected — fetching team info...")
    limits = httpx.Limits(max_connections=15, max_keepalive_connections=15)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=limits) as client:
        sem     = asyncio.Semaphore(15)
        results = await asyncio.gather(
            *[_fetch_team_info(client, sem, mlb_id) for mlb_id in ALL_MLB_TEAMS],
            return_exceptions=True,
        )
    team_info = {}
    for mlb_id, result in zip(ALL_MLB_TEAMS.keys(), results):
        abbr = ALL_MLB_TEAMS[mlb_id][0]
        if isinstance(result, BaseException):
            logger.warning(f"  Team info fetch failed for {abbr}: {result}")
            team_info[mlb_id] = {"name": abbr, "abbreviation": abbr}
        else:
            team_info[mlb_id] = result
    return team_info


async def _fetch_team_info(client, sem, mlb_id: int) -> dict:
    url  = f"https://statsapi.mlb.com/api/v1/teams/{mlb_id}"
    data = await _get_json(client, sem, url)
    t    = data["teams"][0]
    return {"name": t["name"], "abbreviation": t["abbreviation"]}


def _write_divisions_and_teams(db, team_info: dict[int, dict]):
    """Write Division and Team rows. Called only when DB is empty."""
    from database.models import Division

    # Divisions
    division_map = {}
    for name, league, display_name in ALL_DIVISIONS:
        existing = db.query(Division).filter_by(name=name).first()
        if existing:
            division_map[name] = existing.id
            continue
        div = Division(name=name, league=league, display_name=display_name)
        db.add(div)
        db.flush()
        division_map[name] = div.id
        logger.info(f"  Division created: {display_name}")

    # Teams
    existing_mlb_ids = {t.mlb_team_id for t in db.query(Team).all()}
    for mlb_id, (abbr, div_name) in ALL_MLB_TEAMS.items():
        if mlb_id in existing_mlb_ids:
            continue
        info = team_info.get(mlb_id, {})
        team = Team(
            mlb_team_id  = mlb_id,
            name         = info.get("name", abbr),
            abbreviation = info.get("abbreviation", abbr),
            division_id  = division_map[div_name],
            stats_as_of  = date.today(),
            **{k: v for k, v in LEAGUE_AVG_TEAM.items() if hasattr(Team, k)},
        )
        db.add(team)

    db.flush()
    db.commit()
    logger.info(f"  {len(ALL_MLB_TEAMS)} teams created.")


# ─────────────────────────────────────────────────────────────
# ASYNC FETCH PHASE
# ─────────────────────────────────────────────────────────────

async def _fetch_everything(
    our_team_db_id: Optional[int],
    pitcher_rows,
    team_rows,
    known_mlb_player_ids,
) -> dict:
    today             = date.today()
    schedule_end      = today + timedelta(days=SCHEDULE_HORIZON_DAYS)
    game_starts_start = today - timedelta(days=GAME_STARTS_LOOKBACK_DAYS)

    if our_team_db_id is not None:
        mlb_id = DB_TO_MLB_TEAM.get(our_team_db_id)
        schedule_mlb_ids = [mlb_id] if mlb_id is not None else []
    else:
        schedule_mlb_ids = list(MLB_TO_DB_TEAM.keys())

    all_mlb_ids = list(MLB_TO_DB_TEAM.keys())

    limits = httpx.Limits(
        max_connections=MLB_API_CONCURRENCY,
        max_keepalive_connections=MLB_API_CONCURRENCY,
    )

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=limits) as client:
        mlb_sem      = asyncio.Semaphore(MLB_API_CONCURRENCY)
        statcast_sem = asyncio.Semaphore(STATCAST_CONCURRENCY)

        (
            standings_rows,
            schedule_games,
            completed_games,
            pitcher_stats,
            team_stats,
            raw_rosters,
        ) = await asyncio.gather(
            _gather_standings(client, mlb_sem),
            _gather_schedules(client, mlb_sem, schedule_mlb_ids, today, schedule_end),
            _gather_completed_games(client, mlb_sem, all_mlb_ids, game_starts_start, today),
            _gather_pitcher_stats(statcast_sem, pitcher_rows),
            _gather_team_stats(statcast_sem, team_rows),
            _gather_rosters(client, mlb_sem, all_mlb_ids),
        )

        # Boxscores for completed games
        unique_games = {g["mlb_game_id"]: g for g in completed_games}
        boxscore_results = await asyncio.gather(
            *[
                _fetch_starting_pitchers_async(client, mlb_sem, g["mlb_game_id"], g["game_date"])
                for g in unique_games.values()
            ],
            return_exceptions=True,
        )
        starters = []
        for game, result in zip(unique_games.values(), boxscore_results):
            if isinstance(result, BaseException):
                logger.warning(f"Boxscore fetch failed for game {game['mlb_game_id']}: {result}")
                continue
            starters.extend(result)

        # Fetch detail stats for pitchers not yet in the DB.
        # We only need this for new pitchers — existing rows already
        # have stats that _write_pitcher_stats will update via Statcast.
        new_candidates = [
            p for p in raw_rosters
            if p["mlb_player_id"] not in known_mlb_player_ids
        ]

        new_pitcher_details: dict[int, dict] = {}
        if new_candidates:
            logger.info(f"Roster reconcile: {len(new_candidates)} new pitchers found, fetching stats...")
            detail_results = await asyncio.gather(
                *[_fetch_pitcher_detail(client, mlb_sem, p) for p in new_candidates],
                return_exceptions=True,
            )
            for p, result in zip(new_candidates, detail_results):
                if isinstance(result, BaseException):
                    logger.warning(f"Detail fetch failed for {p['name']}: {result}")
                    new_pitcher_details[p["mlb_player_id"]] = {}
                else:
                    new_pitcher_details[p["mlb_player_id"]] = result

    # Attach detail dict to every roster entry (empty for existing pitchers).
    # _reconcile_rosters uses it only when inserting a new row.
    rosters_with_detail = [
        {**p, "detail": new_pitcher_details.get(p["mlb_player_id"], {})}
        for p in raw_rosters
    ]

    return {
        "standings":     standings_rows,
        "schedule":      schedule_games,
        "game_starts":   starters,
        "pitcher_stats": pitcher_stats,
        "team_stats":    team_stats,
        "rosters":       rosters_with_detail,
    }


async def _get_json(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    url: str,
    retries: int = 2,
) -> dict:
    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        async with sem:
            try:
                r = await client.get(url)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise
                last_exc = e
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
        await asyncio.sleep(0.5 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Request failed after {retries} retries with no captured exception: {url}")


# ─────────────────────────────────────────────────────────────
# ROSTER FETCH
#
# Returns ALL pitchers on each team's active roster, not just
# starters. _reconcile_rosters uses _qualifies_as_starter to
# set is_starter correctly, and uses the full list to detect
# who has left the roster entirely (trades, releases, IL).
# ─────────────────────────────────────────────────────────────

async def _gather_rosters(client, sem, all_mlb_ids: list[int]) -> list[dict]:
    results = await asyncio.gather(
        *[_fetch_roster_async(client, sem, mlb_id) for mlb_id in all_mlb_ids],
        return_exceptions=True,
    )
    candidates = []
    for mlb_id, result in zip(all_mlb_ids, results):
        if isinstance(result, BaseException):
            logger.warning(f"Roster fetch failed for mlb_team_id={mlb_id}: {result}")
            continue
        candidates.extend(result)
    return candidates


async def _fetch_roster_async(client, sem, mlb_team_id: int) -> list[dict]:
    """
    Returns all pitchers on the active roster for a team.
    Starter eligibility is determined later in _reconcile_rosters
    so that players who move to the bullpen get is_starter=False
    rather than being silently deactivated.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/teams/{mlb_team_id}/roster"
        f"?rosterType=active&season={date.today().year}"
        f"&hydrate=person(stats(type=season,group=pitching))"
    )
    data = await _get_json(client, sem, url)
    pitchers = []
    for entry in data.get("roster", []):
        if entry.get("position", {}).get("abbreviation") != "P":
            continue
        person       = entry.get("person", {})
        stats_list   = person.get("stats", [])
        season_stats = {}
        for s in stats_list:
            if s.get("type", {}).get("displayName") == "season":
                splits = s.get("splits", [])
                if splits:
                    season_stats = splits[0].get("stat", {})
                break

        gs = int(season_stats.get("gamesStarted", 0))
        gp = int(season_stats.get("gamesPitched", 0)) or gs
        ip = _parse_ip(season_stats.get("inningsPitched", "0"))
        bf = int(season_stats.get("battersFaced", max(int(ip * 4), 1)))

        pitchers.append({
            "mlb_player_id": person["id"],
            "name":          person.get("fullName", str(person["id"])),
            "throws":        person.get("pitchHand", {}).get("code", "R"),
            "team_mlb_id":   mlb_team_id,
            "games_started": gs,
            "ip":            ip,
            "strikeouts":    int(season_stats.get("strikeOuts", 0)),
            "walks":         int(season_stats.get("baseOnBalls", 0)),
            "batters_faced": bf,
            "gp":            gp,
            # Starter flag resolved here so _reconcile_rosters can
            # update is_starter without re-running the eligibility math.
            "is_starter":    _qualifies_as_starter(gs, gp, ip),
        })
    return pitchers


async def _fetch_pitcher_detail(client, sem, p: dict) -> dict:
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{p['mlb_player_id']}/stats"
        f"?stats=season&group=pitching&season={date.today().year}"
    )
    data   = await _get_json(client, sem, url)
    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return {}
    s  = splits[0].get("stat", {})
    bf = p["batters_faced"]
    try:
        obp = float(s.get("obp", 0) or 0)
        slg = float(s.get("slg", 0) or 0)
        ops = (obp + slg) if (obp + slg) > 0.3 else LEAGUE_AVG_PITCHER["ops_allowed"]
    except (ValueError, TypeError):
        ops = LEAGUE_AVG_PITCHER["ops_allowed"]
    return {
        "era":         float(s.get("era", "4.50") or 4.50),
        "ops_allowed": ops,
        "k_pct":       p["strikeouts"] / max(bf, 1),
        "bb_pct":      p["walks"]      / max(bf, 1),
    }


# ─────────────────────────────────────────────────────────────
# ROSTER RECONCILE
#
# Runs every sync. Compares the full current MLB active roster
# snapshot against DB state and:
#   - Inserts new pitchers (never seen before)
#   - Updates team_id on trades
#   - Updates is_starter when a pitcher moves to/from the bullpen
#   - Reactivates pitchers who return from IL or minors
#   - Deactivates pitchers no longer on any active roster
#
# Never deletes rows — all GameStart, fatigue, and optimizer
# history is preserved and linked by pitcher.id.
# ─────────────────────────────────────────────────────────────

def _reconcile_rosters(db, roster_candidates: list[dict] | None):
    if not roster_candidates:
        return

    today = date.today()

    # Build lookup of all DB pitchers by mlb_player_id
    all_pitchers = {p.mlb_player_id: p for p in db.query(Pitcher).all()}

    # Set of mlb_player_ids currently on an active MLB roster
    active_mlb_ids = {p["mlb_player_id"] for p in roster_candidates}

    # ── Deactivate pitchers no longer on any active roster ─────
    deactivated = 0
    for mlb_id, pitcher in all_pitchers.items():
        if pitcher.is_active and mlb_id not in active_mlb_ids:
            pitcher.is_active  = False
            pitcher.is_starter = False
            logger.info(f"  Deactivated: {pitcher.name} (no longer on active roster)")
            deactivated += 1

    # ── Insert or update ───────────────────────────────────────
    inserted = updated = 0
    for candidate in roster_candidates:
        mlb_id     = candidate["mlb_player_id"]
        db_team_id = MLB_TO_DB_TEAM.get(candidate["team_mlb_id"])
        if db_team_id is None:
            continue

        if mlb_id not in all_pitchers:
            # Brand new pitcher — insert with league-average defaults
            _insert_pitcher(db, candidate, db_team_id, today)
            inserted += 1
        else:
            pitcher = all_pitchers[mlb_id]
            changed = False

            # Trade detected — update team
            if pitcher.team_id != db_team_id:
                logger.info(f"  Trade: {pitcher.name} → team_id {db_team_id}")
                pitcher.team_id = db_team_id
                changed = True

            # Starter status changed (promoted, demoted, role change)
            if pitcher.is_starter != candidate["is_starter"]:
                pitcher.is_starter = candidate["is_starter"]
                action = "promoted to starter" if candidate["is_starter"] else "moved to bullpen"
                logger.info(f"  Role change: {pitcher.name} {action}")
                changed = True

            # Reactivate if previously marked inactive (returned from IL/minors)
            if not pitcher.is_active:
                pitcher.is_active = True
                logger.info(f"  Reactivated: {pitcher.name}")
                changed = True

            if changed:
                pitcher.stats_as_of = today
                updated += 1

    db.flush()
    logger.info(
        f"Roster reconcile: {inserted} inserted, {updated} updated, "
        f"{deactivated} deactivated."
    )


def _insert_pitcher(db, p: dict, db_team_id: int, today: date):
    """Insert a brand-new pitcher row with league-average stat defaults."""
    detail = p.get("detail", {})
    ops    = detail.get("ops_allowed") or LEAGUE_AVG_PITCHER["ops_allowed"]
    pitcher = Pitcher(
        mlb_player_id      = p["mlb_player_id"],
        name               = p["name"],
        team_id            = db_team_id,
        throws             = p["throws"],
        is_starter         = p["is_starter"],
        is_active          = True,
        stats_as_of        = today,
        era                = min(float(detail.get("era") or 4.50), 14.99),
        k_pct              = detail.get("k_pct")  or LEAGUE_AVG_PITCHER["k_pct"],
        bb_pct             = detail.get("bb_pct") or LEAGUE_AVG_PITCHER["bb_pct"],
        k_pct_60d          = detail.get("k_pct")  or LEAGUE_AVG_PITCHER["k_pct"],
        bb_pct_60d         = detail.get("bb_pct") or LEAGUE_AVG_PITCHER["bb_pct"],
        ops_allowed        = ops,
        ops_allowed_60d    = ops,
        ops_allowed_vs_rhb = ops - 0.020,
        ops_allowed_vs_lhb = (ops + 0.025) if p["throws"] == "R" else (ops - 0.010),
        woba_allowed       = LEAGUE_AVG_PITCHER["woba_allowed"],
        fip                = LEAGUE_AVG_PITCHER["fip"],
        xfip               = LEAGUE_AVG_PITCHER["xfip"],
        xfip_60d           = LEAGUE_AVG_PITCHER["xfip"],
        fb_velo            = LEAGUE_AVG_PITCHER["fb_velo"],
        spin_rate          = LEAGUE_AVG_PITCHER["spin_rate"],
        extension          = LEAGUE_AVG_PITCHER["extension"],
        whiff_pct          = LEAGUE_AVG_PITCHER["whiff_pct"],
        zone_pct           = LEAGUE_AVG_PITCHER["zone_pct"],
        chase_pct          = LEAGUE_AVG_PITCHER["chase_pct"],
        fastball_usage     = LEAGUE_AVG_PITCHER["fastball_usage"],
        breaking_usage     = LEAGUE_AVG_PITCHER["breaking_usage"],
        offspeed_usage     = LEAGUE_AVG_PITCHER["offspeed_usage"],
        hard_hit_pct       = LEAGUE_AVG_PITCHER["hard_hit_pct"],
        xwoba              = LEAGUE_AVG_PITCHER["xwoba"],
    )
    db.add(pitcher)
    gs = p.get("games_started", 0)
    ip = p.get("ip", 0.0)
    logger.info(
        f"  New pitcher: {p['name']} "
        f"({'starter' if p['is_starter'] else 'reliever'}, "
        f"{gs} GS, {ip:.1f} IP)"
    )


# ─────────────────────────────────────────────────────────────
# STANDINGS
# ─────────────────────────────────────────────────────────────

async def _gather_standings(client, sem) -> list[dict]:
    results = await asyncio.gather(
        *[_fetch_standings_async(client, sem, lid) for lid in (103, 104)],
        return_exceptions=True,
    )
    rows = []
    for league_id, result in zip((103, 104), results):
        if isinstance(result, BaseException):
            logger.error(f"fetch_standings failed for league {league_id}: {result}")
            continue
        rows.extend(result)
    return rows


async def _fetch_standings_async(client, sem, league_id: int) -> list[dict]:
    url = (
        f"https://statsapi.mlb.com/api/v1/standings"
        f"?leagueId={league_id}&season={date.today().year}&standingsTypes=regularSeason"
    )
    data = await _get_json(client, sem, url)
    rows = []
    for record in data.get("records", []):
        division_id   = record.get("division", {}).get("id")
        division_name = (
            MLB_DIVISION_ID_TO_NAME.get(division_id)
            or record.get("division", {}).get("nameShort")
            or record.get("division", {}).get("name")
            or f"UNKNOWN_{division_id}"
        )
        for team_rec in record.get("teamRecords", []):
            team   = team_rec.get("team", {})
            mlb_id = team.get("id")
            rows.append({
                "mlb_team_id":      mlb_id,
                "team_name":        team.get("name") or team.get("teamName") or str(mlb_id),
                "division":         division_name,
                "wins":             team_rec.get("wins", 0),
                "losses":           team_rec.get("losses", 0),
                "games_behind":     float(team_rec.get("gamesBack", "0").replace("-", "0")),
                "win_pct":          float(team_rec.get("winningPercentage", "0.000")),
                "run_differential": team_rec.get("runDifferential", 0),
            })
    return rows


def _write_standings(db, rows: list[dict] | None):
    if not rows:
        return
    today = date.today()
    for row in rows:
        mlb_team_id = row.get("mlb_team_id")
        if mlb_team_id is None:
            continue
        db_team_id = MLB_TO_DB_TEAM.get(mlb_team_id)
        if not db_team_id:
            continue
        team = db.query(Team).filter_by(id=db_team_id).first()
        if not team:
            continue
        existing = db.query(StandingsSnapshot).filter_by(
            team_id=db_team_id, snapshot_date=today
        ).first()
        kwargs = dict(
            wins             = row["wins"],
            losses           = row["losses"],
            games_behind     = row["games_behind"],
            win_pct          = row["win_pct"],
            run_differential = row.get("run_differential", 0),
            ops              = team.ops or 0.720,
        )
        if existing:
            for k, v in kwargs.items():
                setattr(existing, k, v)
        else:
            db.add(StandingsSnapshot(
                team_id       = db_team_id,
                snapshot_date = today,
                season        = today.year,
                division      = row["division"],
                team_name     = row["team_name"],
                **kwargs,
            ))


# ─────────────────────────────────────────────────────────────
# SCHEDULE
# ─────────────────────────────────────────────────────────────

async def _gather_schedules(client, sem, mlb_ids: list[int], start: date, end: date) -> list[dict]:
    results = await asyncio.gather(
        *[_fetch_schedule_async(client, sem, mlb_id, start, end) for mlb_id in mlb_ids],
        return_exceptions=True,
    )
    all_games: dict[int, dict] = {}
    for mlb_id, result in zip(mlb_ids, results):
        if isinstance(result, BaseException):
            logger.error(f"fetch_schedule failed for mlb_team_id={mlb_id}: {result}")
            continue
        for g in result:
            all_games.setdefault(g["mlb_game_id"], g)
    return list(all_games.values())


async def _fetch_schedule_async(client, sem, mlb_team_id: int, start: date, end: date) -> list[dict]:
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&teamId={mlb_team_id}"
        f"&startDate={start.isoformat()}&endDate={end.isoformat()}"
        f"&gameType=R&hydrate=team"
    )
    data  = await _get_json(client, sem, url)
    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            teams  = game.get("teams", {})
            home   = teams.get("home", {}).get("team", {})
            away   = teams.get("away", {}).get("team", {})
            status = game.get("status", {}).get("detailedState", "")
            games.append({
                "mlb_game_id":     game.get("gamePk"),
                "game_date":       date.fromisoformat(date_entry["date"]),
                "home_mlb_id":     home.get("id"),
                "away_mlb_id":     away.get("id"),
                "home_name":       home.get("name") or home.get("teamName", ""),
                "away_name":       away.get("name") or away.get("teamName", ""),
                "series_game_num": game.get("seriesGameNumber", 1),
                "h2h_remaining":   game.get("gamesInSeries", 3) - game.get("seriesGameNumber", 1),
                "status":          status,
            })
    return games


def _write_schedule(db, games: list[dict] | None):
    if not games:
        return
    existing = {row.mlb_game_id: row for row in db.query(Schedule).all()}
    inserted = updated = skipped = 0
    for g in games:
        game_id      = g["mlb_game_id"]
        is_completed = g.get("status") == "Final"
        if game_id in existing:
            row = existing[game_id]
            if is_completed and not row.game_completed:
                row.game_completed = True
                updated += 1
            else:
                skipped += 1
            continue
        home_db = MLB_TO_DB_TEAM.get(g["home_mlb_id"])
        away_db = MLB_TO_DB_TEAM.get(g["away_mlb_id"])
        if home_db is None and away_db is None:
            continue
        db.add(Schedule(
            mlb_game_id     = game_id,
            game_date       = g["game_date"],
            home_team_id    = home_db or away_db,
            away_team_id    = away_db or home_db,
            is_home         = False,
            series_game_num = min(g.get("series_game_num", 1), 4),
            h2h_remaining   = g.get("h2h_remaining", 0),
            game_completed  = is_completed,
        ))
        inserted += 1
    db.flush()
    logger.info(f"Schedule: {inserted} inserted, {updated} marked complete, {skipped} unchanged.")


# ─────────────────────────────────────────────────────────────
# PITCHER + TEAM STATS
# ─────────────────────────────────────────────────────────────

async def _gather_pitcher_stats(sem, pitcher_rows: list[tuple]) -> list[tuple]:
    async def fetch_one(pitcher_db_id, mlb_player_id, name):
        async with sem:
            try:
                stats = await asyncio.to_thread(fetch_pitcher_statcast, mlb_player_id, 60)
                return (pitcher_db_id, name, stats, None)
            except Exception as e:
                return (pitcher_db_id, name, None, e)
    if not pitcher_rows:
        return []
    return await asyncio.gather(*[fetch_one(*row) for row in pitcher_rows])


async def _gather_team_stats(sem, team_rows: list[tuple]) -> list[tuple]:
    async def fetch_one(team_db_id, mlb_team_id, name):
        async with sem:
            try:
                stats = await asyncio.to_thread(fetch_team_statcast, mlb_team_id, 60)
                return (team_db_id, name, stats, None)
            except Exception as e:
                return (team_db_id, name, None, e)
    if not team_rows:
        return []
    return await asyncio.gather(*[fetch_one(*row) for row in team_rows])


def _write_pitcher_stats(db, results: list[tuple] | None):
    if not results:
        return
    pitcher_by_id = {
        p.id: p for p in db.query(Pitcher).filter_by(is_starter=True, is_active=True).all()
    }
    updated = 0
    for pitcher_db_id, name, stats, error in results:
        if error:
            logger.warning(f"  Failed pitcher {name}: {error}")
            continue
        if not stats:
            continue
        pitcher = pitcher_by_id.get(pitcher_db_id)
        if not pitcher:
            continue
        for col, val in stats.items():
            if hasattr(pitcher, col) and val is not None and val == val:
                setattr(pitcher, col, float(val))
        pitcher.stats_as_of = date.today()
        updated += 1
    logger.info(f"Pitcher stats: {updated}/{len(results)} updated.")


def _write_team_stats(db, results: list[tuple] | None):
    if not results:
        return
    team_by_id = {t.id: t for t in db.query(Team).all()}
    updated = 0
    for team_db_id, name, stats, error in results:
        if error:
            logger.warning(f"  Failed team {name}: {error}")
            continue
        if not stats:
            continue
        team = team_by_id.get(team_db_id)
        if not team:
            continue
        for col, val in stats.items():
            if hasattr(team, col) and val is not None and val == val:
                setattr(team, col, float(val))
        team.stats_as_of = date.today()
        updated += 1
    logger.info(f"Team stats: {updated}/{len(results)} updated.")


# ─────────────────────────────────────────────────────────────
# GAME STARTS
# ─────────────────────────────────────────────────────────────

async def _gather_completed_games(client, sem, mlb_ids: list[int], start: date, end: date) -> list[dict]:
    results = await asyncio.gather(
        *[_fetch_completed_games_async(client, sem, mlb_id, start, end) for mlb_id in mlb_ids],
        return_exceptions=True,
    )
    games = []
    for mlb_id, result in zip(mlb_ids, results):
        if isinstance(result, BaseException):
            logger.warning(f"  Failed fetching completed games for mlb_team_id={mlb_id}: {result}")
            continue
        games.extend(result)
    return games


async def _fetch_completed_games_async(client, sem, mlb_team_id: int, start: date, end: date) -> list[dict]:
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&teamId={mlb_team_id}"
        f"&startDate={start.isoformat()}&endDate={end.isoformat()}"
        f"&gameType=R"
    )
    data  = await _get_json(client, sem, url)
    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("detailedState", "") != "Final":
                continue
            games.append({
                "mlb_game_id": game.get("gamePk"),
                "game_date":   date.fromisoformat(date_entry["date"]),
            })
    return games


async def _fetch_starting_pitchers_async(client, sem, game_pk: int, game_date: date) -> list[dict]:
    url        = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    data       = await _get_json(client, sem, url)
    teams_data = data.get("teams", {})
    results    = []
    for side in ("home", "away"):
        team_block      = teams_data.get(side, {})
        players         = team_block.get("players", {})
        opponent_side   = "away" if side == "home" else "home"
        opponent_mlb_id = teams_data.get(opponent_side, {}).get("team", {}).get("id")
        for _, player_data in players.items():
            stats = player_data.get("stats", {}).get("pitching", {})
            if not stats or stats.get("gamesStarted", 0) != 1:
                continue
            person      = player_data.get("person", {})
            pitch_count = stats.get("numberOfPitches", 0) or 0
            ip          = _parse_innings_boxscore(stats.get("inningsPitched", "0.0"))
            strikeouts  = stats.get("strikeOuts", 0) or 0
            walks       = stats.get("baseOnBalls", 0) or 0
            hits        = stats.get("hits", 0) or 0
            earned_runs = stats.get("earnedRuns", 0) or 0
            results.append({
                "mlb_player_id":   person.get("id"),
                "game_date":       game_date,
                "opponent_mlb_id": opponent_mlb_id,
                "pitch_count":     int(pitch_count),
                "innings_pitched": ip,
                "game_score":      _compute_game_score(ip, hits, earned_runs, walks, strikeouts),
            })
            break
    return results


def _write_game_starts(db, starters: list[dict] | None):
    if not starters:
        return
    from database.models import Pitcher, GameStart
    pitcher_by_mlb_id = {p.mlb_player_id: p for p in db.query(Pitcher).all()}
    if not pitcher_by_mlb_id:
        logger.warning("No pitchers in DB — skipping game starts sync")
        return
    existing_keys = {
        (row.pitcher_id, row.game_date)
        for row in db.query(GameStart.pitcher_id, GameStart.game_date).all()
    }
    written = 0
    for outing in starters:
        pitcher = pitcher_by_mlb_id.get(outing["mlb_player_id"])
        if not pitcher:
            continue
        key = (pitcher.id, outing["game_date"])
        if key in existing_keys:
            continue
        opponent_db_id = MLB_TO_DB_TEAM.get(outing["opponent_mlb_id"])
        if opponent_db_id is None:
            continue
        db.add(GameStart(
            pitcher_id       = pitcher.id,
            game_date        = outing["game_date"],
            opponent_team_id = opponent_db_id,
            pitch_count      = outing["pitch_count"],
            innings_pitched  = outing["innings_pitched"],
            game_score       = outing["game_score"],
            velocity_delta   = 0.0,
            was_recommended  = None,
        ))
        existing_keys.add(key)
        written += 1
    db.flush()
    logger.info(f"Game starts: {len(starters)} outings checked, {written} new rows written.")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _qualifies_as_starter(gs: int, gp: int, ip: float) -> bool:
    """
    True if pitcher meets starter criteria:
      - At least STARTER_MIN_GS games started
      - Averages at least STARTER_MIN_IP_PER_APP innings per appearance
        (uses total appearances, not just starts, as the denominator)

    Using gamesPitched (gp) rather than gamesStarted (gs) as the
    denominator prevents false positives for two-way players or swingmen
    who have a few starts but many more relief appearances. Their total IP
    is spread across all outings, not just the starts.

    Filters out:
      - Relievers with a single spot start (gs < STARTER_MIN_GS)
      - Openers/bulk-relievers who start but exit after 2 innings
        before a bulk arm takes over (low IP/appearance ratio)
    """
    if gs < STARTER_MIN_GS:
        return False
    denominator = max(gp, gs)
    return (ip / denominator) >= STARTER_MIN_IP_PER_APP


def _parse_ip(ip_str) -> float:
    """
    Handles both season-total decimal IP ('43.0', '60.1') and
    per-game traditional notation ('6.2' = 6 innings + 2 outs).
    MLB API returns decimal for season totals, traditional for game logs.
    """
    try:
        val  = float(str(ip_str))
        frac = val - int(val)
        if frac <= 0.29:
            return int(val) + (frac * 10 / 3)
        return val
    except (TypeError, ValueError):
        return 0.0


def _parse_innings_boxscore(ip_str) -> float:
    """Boxscore always uses traditional notation: '6.1' = 6⅓ innings."""
    try:
        whole, _, frac = str(ip_str).partition(".")
        frac_map = {"0": 0.0, "1": 1/3, "2": 2/3}
        return float(whole or 0) + frac_map.get(frac, 0.0)
    except Exception:
        return 0.0


def _compute_game_score(ip: float, hits: int, earned_runs: int, walks: int, strikeouts: int) -> int:
    score  = 50
    score += int(ip * 2)
    score += strikeouts
    score -= hits * 2
    score -= earned_runs * 2
    score -= walks * 2
    return int(np.clip(score, 0, 100))