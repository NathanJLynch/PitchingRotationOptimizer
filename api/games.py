# api/games.py
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from typing import cast as tcast

from database.models import (
    Schedule, Team, Pitcher, OptimizerRun, GameStart,
    SessionFactory, get_db_session,
)
from optimizer.dp_engine import solve, Game, OptimizationResult
from optimizer.fatigue_model import update_fatigue, build_fatigue_state
from optimizer.dp_engine import discretize_fatigue
from optimizer.matchup_model import PitcherProfile, OpponentProfile
from api.pitchers import _pitcher_to_profile, _team_to_profile


router = APIRouter(prefix="/games", tags=["games"])


# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────

class GameSummary(BaseModel):
    game_id:                  int
    game_date:                date
    opponent_team_id:         int
    opponent_name:            str
    is_home:                  bool
    series_game_num:          Optional[int]
    scheduled_pitcher_id:     Optional[int]
    scheduled_pitcher_name:   Optional[str]
    game_completed:           bool


class UpcomingGamesResponse(BaseModel):
    team_id: int
    games:   list[GameSummary]


class OptimizeRequest(BaseModel):
    team_id:      int
    horizon_days: int = Field(default=14, ge=3, le=45,
                              description="How many days ahead to optimize the rotation for")
    start_date:   Optional[date] = Field(default=None,
                              description="Defaults to today")
    trigger:      str = Field(default="manual",
                              description="'scheduled' | 'injury' | 'manual'")
    notes:        Optional[str] = None


class AssignmentResponse(BaseModel):
    game_id:           int
    game_date:         date
    opponent_team_id:  int
    opponent_name:     str
    pitcher_id:        int
    pitcher_name:      str
    fatigue_level:     int = Field(..., ge=0, le=4)
    matchup_score:     float
    division_bonus:    float
    fatigue_penalty:   float
    total_score:       float
    division_regime:   str
    score_breakdown:   dict


class OptimizeResponse(BaseModel):
    optimizer_run_id: int
    team_id:          int
    run_date:         date
    horizon_start:    date
    horizon_end:      date
    total_score:      float
    assignments:      list[AssignmentResponse]


class GameResultRequest(BaseModel):
    pitcher_id:           int
    actual_pitches:       int   = Field(..., ge=1, le=150)
    actual_innings:       float = Field(..., ge=0)
    velocity_delta:       float = Field(default=0.0)
    k_pct_actual:         Optional[float] = None
    ops_allowed_actual:   Optional[float] = None
    wpa:                  Optional[float] = None
    home_score:           Optional[int]   = None
    away_score:           Optional[int]   = None


class GameResultResponse(BaseModel):
    game_id:                int
    pitcher_id:             int
    updated_fatigue:        dict
    matched_recommendation: Optional[bool]


# ─────────────────────────────────────────────────────────────
# DEPENDENCY
# ─────────────────────────────────────────────────────────────

def get_db():
    with get_db_session(SessionFactory) as db:
        yield db


# ─────────────────────────────────────────────────────────────
# SHARED HELPER (extracted from get_optimizer_run)
# ─────────────────────────────────────────────────────────────

def _fetch_optimizer_run(run_id: int, db) -> OptimizeResponse:
    """
    Core logic for retrieving a persisted optimizer run.
    Shared by get_optimizer_run and get_latest_assignments
    to avoid calling a route handler from another route handler.
    """
    run = db.query(OptimizerRun).filter_by(id=run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"OptimizerRun {run_id} not found")

    # ── Batch-load pitchers and schedule rows to avoid N+1 queries ──
    pitcher_ids     = [a["pitcher_id"] for a in run.assignments_json]
    game_ids        = [a["game_id"]    for a in run.assignments_json]

    pitchers_map    = {
        p.id: p for p in db.query(Pitcher).filter(Pitcher.id.in_(pitcher_ids)).all()
    }
    schedule_map    = {
        r.id: r for r in db.query(Schedule).filter(Schedule.id.in_(game_ids)).all()
    }

    assignment_responses = []
    for a in run.assignments_json:
        pitcher      = pitchers_map.get(a["pitcher_id"])
        schedule_row = schedule_map.get(a["game_id"])

        opponent_id   = None
        opponent_name = "Unknown"
        if schedule_row:
            is_home     = (schedule_row.home_team_id == run.our_team_id)
            opponent_id = schedule_row.away_team_id if is_home else schedule_row.home_team_id
            opponent    = db.query(Team).filter_by(id=opponent_id).first()
            opponent_name = opponent.name if opponent else "Unknown"

        assignment_responses.append(AssignmentResponse(
            game_id          = a["game_id"],
            game_date        = date.fromisoformat(a["game_date"]),
            opponent_team_id = opponent_id or 0,
            opponent_name    = opponent_name,
            pitcher_id       = a["pitcher_id"],
            pitcher_name     = pitcher.name if pitcher else "Unknown",
            fatigue_level    = a["fatigue_level"],
            matchup_score    = a["matchup_score"],
            division_bonus   = a["division_bonus"],
            fatigue_penalty  = a["fatigue_penalty"],
            total_score      = a["total_score"],
            division_regime  = (
                schedule_row.score_breakdown.get("division_regime", "unknown")
                if schedule_row and schedule_row.score_breakdown else "unknown"
            ),
            score_breakdown  = schedule_row.score_breakdown if schedule_row else {},
        ))

    return OptimizeResponse(
        optimizer_run_id = run.id,
        team_id          = run.our_team_id,
        run_date         = run.run_date,
        horizon_start    = run.horizon_start,
        horizon_end      = run.horizon_end,
        total_score      = run.total_score,
        assignments      = assignment_responses,
    )


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@router.get("/upcoming", response_model=UpcomingGamesResponse)
def get_upcoming_games(
    team_id: int = Query(...),
    days:    int = Query(default=10, ge=1, le=60),
    db=Depends(get_db),
):
    """
    Returns the upcoming schedule for a team with opponent info
    and any existing optimizer-recommended starter.
    """
    today    = date.today()
    end_date = today + timedelta(days=days)

    rows = (
        db.query(Schedule)
        .filter(
            ((Schedule.home_team_id == team_id) | (Schedule.away_team_id == team_id)),
            Schedule.game_date >= today,
            Schedule.game_date <= end_date,
        )
        .order_by(Schedule.game_date.asc())
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduled games found for team {team_id} in next {days} days"
        )

    # Batch-load all opponents and scheduled pitchers
    opponent_ids  = {
        (r.away_team_id if r.home_team_id == team_id else r.home_team_id) for r in rows
    }
    pitcher_ids   = {r.scheduled_pitcher_id for r in rows if r.scheduled_pitcher_id}

    opponents_map = {
        t.id: t for t in db.query(Team).filter(Team.id.in_(opponent_ids)).all()
    }
    pitchers_map  = {
        p.id: p for p in db.query(Pitcher).filter(Pitcher.id.in_(pitcher_ids)).all()
    }

    summaries = []
    for row in rows:
        is_home     = (row.home_team_id == team_id)
        opponent_id = row.away_team_id if is_home else row.home_team_id
        opponent    = opponents_map.get(opponent_id)

        pitcher      = pitchers_map.get(row.scheduled_pitcher_id) if row.scheduled_pitcher_id else None
        pitcher_name = pitcher.name if pitcher else None

        summaries.append(GameSummary(
            game_id                = row.id,
            game_date              = row.game_date,
            opponent_team_id       = opponent_id,
            opponent_name          = opponent.name if opponent else "Unknown",
            is_home                = is_home,
            series_game_num        = row.series_game_num,
            scheduled_pitcher_id   = row.scheduled_pitcher_id,
            scheduled_pitcher_name = pitcher_name,
            game_completed         = row.game_completed,
        ))

    return UpcomingGamesResponse(team_id=team_id, games=summaries)


@router.post("/optimizer/run", response_model=OptimizeResponse)
def run_optimizer(
    request: OptimizeRequest,
    db=Depends(get_db),
):
    """
    Triggers a full DP solve over the given horizon.

    Pipeline:
      1. Pull upcoming games for the team within the horizon
      2. Pull active starting pitchers for the team
      3. Build Game and PitcherProfile objects
      4. Run dp_engine.solve()
      5. Persist results to OptimizerRun + update Schedule rows
      6. Return the full assignment plan
    """
    start_date = request.start_date or date.today()
    end_date   = start_date + timedelta(days=request.horizon_days)

    # ── 1. Load games ──────────────────────────────────────────
    schedule_rows = (
        db.query(Schedule)
        .filter(
            ((Schedule.home_team_id == request.team_id) | (Schedule.away_team_id == request.team_id)),
            Schedule.game_date >= start_date,
            Schedule.game_date <= end_date,
            Schedule.game_completed == False,
        )
        .order_by(Schedule.game_date.asc())
        .all()
    )

    if not schedule_rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No upcoming games found for team {request.team_id} "
                f"between {start_date} and {end_date}"
            )
        )

    # ── 2. Load active starters ────────────────────────────────
    pitcher_rows = (
        db.query(Pitcher)
        .filter_by(team_id=request.team_id, is_starter=True, is_active=True)
        .all()
    )

    if not pitcher_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No active starting pitchers found for team {request.team_id}"
        )

    # Build PitcherProfile objects — fatigue state included at construction time
    pitchers = [
        _pitcher_to_profile(p, db_session=db, as_of_date=start_date)
        for p in pitcher_rows
    ]

    # ── 3. Build Game objects ──────────────────────────────────
    opponent_ids  = {
        (r.away_team_id if r.home_team_id == request.team_id else r.home_team_id)
        for r in schedule_rows
    }
    opponents_map = {
        t.id: t for t in db.query(Team).filter(Team.id.in_(opponent_ids)).all()
    }

    games = []
    for row in schedule_rows:
        is_home     = (row.home_team_id == request.team_id)
        opponent_id = row.away_team_id if is_home else row.home_team_id
        opponent    = opponents_map.get(opponent_id)

        if not opponent:
            continue   # skip games with missing opponent data

        games.append(Game(
            game_id          = row.id,
            date             = row.game_date,
            opponent_team_id = opponent_id,
            opponent         = _team_to_profile(opponent),
            is_home          = is_home,
            series_game_num  = row.series_game_num or 1,
            h2h_remaining    = row.h2h_remaining or 0,
        ))

    if not games:
        raise HTTPException(
            status_code=422,
            detail="No games could be built — opponent data missing for all scheduled games"
        )

    # ── 4. Run the solver ──────────────────────────────────────
    try:
        result: OptimizationResult = solve(
            games       = games,
            pitchers    = pitchers,
            our_team_id = request.team_id,
            db_session  = db,
            run_date    = start_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── 5. Persist OptimizerRun ────────────────────────────────
    optimizer_run = OptimizerRun(
        our_team_id      = request.team_id,
        run_date         = start_date,
        horizon_start    = games[0].date,
        horizon_end      = games[-1].date,
        horizon_days     = result.horizon_days,
        total_score      = result.total_score,
        assignments_json = [
            {
                "game_id":         a.game.game_id,
                "game_date":       a.game.date.isoformat(),
                "pitcher_id":      a.pitcher.id,
                "fatigue_level":   a.fatigue_level,
                "matchup_score":   a.matchup_score,
                "division_bonus":  a.division_bonus,
                "fatigue_penalty": a.fatigue_penalty,
                "total_score":     a.total_score,
            }
            for a in result.assignments
        ],
        dp_metadata = {
            "n_games":    len(games),
            "n_pitchers": len(pitchers),
        },
        trigger = request.trigger,
        notes   = request.notes,
    )
    db.add(optimizer_run)
    db.flush()   # get optimizer_run.id before linking schedule rows

    # ── Update Schedule rows + build response ──────────────────
    schedule_map  = {r.id: r for r in schedule_rows}
    pitcher_id_to_name = {p.id: p.name for p in pitcher_rows}

    # Batch-load opponents for response construction
    result_opponent_ids = {a.game.opponent_team_id for a in result.assignments}
    result_opponents    = {
        t.id: t for t in db.query(Team).filter(Team.id.in_(result_opponent_ids)).all()
    }

    assignment_responses = []
    for a in result.assignments:
        row      = schedule_map.get(a.game.game_id)
        opponent = result_opponents.get(a.game.opponent_team_id)

        if row:
            row.scheduled_pitcher_id = a.pitcher.id
            row.optimizer_run_id     = optimizer_run.id
            row.optimizer_score      = a.total_score
            row.score_breakdown      = a.score_breakdown

        assignment_responses.append(AssignmentResponse(
            game_id          = a.game.game_id,
            game_date        = a.game.date,
            opponent_team_id = a.game.opponent_team_id,
            opponent_name    = opponent.name if opponent else "Unknown",
            pitcher_id       = a.pitcher.id,
            pitcher_name     = pitcher_id_to_name.get(a.pitcher.id, "Unknown"),
            fatigue_level    = a.fatigue_level,
            matchup_score    = round(a.matchup_score, 4),
            division_bonus   = round(a.division_bonus, 4),
            fatigue_penalty  = round(a.fatigue_penalty, 4),
            total_score      = round(a.total_score, 4),
            division_regime  = a.score_breakdown.get("division_regime", "unknown"),
            score_breakdown  = a.score_breakdown,
        ))

    db.commit()

    return OptimizeResponse(
        optimizer_run_id = tcast(int, optimizer_run.id),
        team_id          = request.team_id,
        run_date         = start_date,
        horizon_start    = result.assignments[0].game.date,
        horizon_end      = result.assignments[-1].game.date,
        total_score      = round(result.total_score, 4),
        assignments      = assignment_responses,
    )


@router.get("/optimizer/runs/{run_id}", response_model=OptimizeResponse)
def get_optimizer_run(run_id: int, db=Depends(get_db)):
    """
    Retrieve a previously computed optimizer run by ID.
    Useful for reviewing past recommendations without re-solving.
    """
    return _fetch_optimizer_run(run_id, db)


@router.get("/optimizer/assignments", response_model=OptimizeResponse)
def get_latest_assignments(
    team_id: int = Query(...),
    db=Depends(get_db),
):
    """
    Returns the most recent optimizer run for a team.
    The dashboard polls this endpoint to display the current
    recommended rotation without triggering a new solve.
    """
    run = (
        db.query(OptimizerRun)
        .filter_by(our_team_id=team_id)
        .order_by(OptimizerRun.run_date.desc(), OptimizerRun.created_at.desc())
        .first()
    )
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"No optimizer runs found for team {team_id}. Run /optimizer/run first."
        )

    return _fetch_optimizer_run(run.id, db)


@router.post("/{game_id}/result", response_model=GameResultResponse)
def record_game_result(
    game_id:          int,
    result:           GameResultRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
):
    """
    Records the actual outcome of a start after the game completes.

    This is the feedback loop:
      1. Writes the actual outing to GameStart (via update_fatigue)
      2. Updates the Schedule row with final score + actual pitcher
      3. Flags whether the actual pitcher matched the recommendation
         (used downstream for model validation)

    fatigue_model.update_fatigue() handles the upsert and
    recomputes the pitcher's FatigueState for future optimizer runs.
    """
    schedule_row = db.query(Schedule).filter_by(id=game_id).first()
    if not schedule_row:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    pitcher = db.query(Pitcher).filter_by(id=result.pitcher_id).first()
    if not pitcher:
        raise HTTPException(status_code=404, detail=f"Pitcher {result.pitcher_id} not found")

    # ── Update fatigue state (writes GameStart row) ────────────
    new_state = update_fatigue(
        pitcher_id     = result.pitcher_id,
        game_date      = schedule_row.game_date,
        actual_pitches = result.actual_pitches,
        actual_innings = result.actual_innings,
        velocity_delta = result.velocity_delta,
        db_session     = db,
    )

    # ── Fill in additional GameStart fields ────────────────────
    game_start = (
        db.query(GameStart)
        .filter_by(pitcher_id=result.pitcher_id, game_date=schedule_row.game_date)
        .first()
    )
    matched_recommendation = None
    if game_start:
        opponent_id = (
            schedule_row.away_team_id
            if schedule_row.home_team_id == pitcher.team_id
            else schedule_row.home_team_id
        )
        game_start.schedule_id        = schedule_row.id
        game_start.opponent_team_id   = opponent_id
        game_start.k_pct_actual       = result.k_pct_actual
        game_start.ops_allowed_actual = result.ops_allowed_actual
        game_start.wpa                = result.wpa

        matched_recommendation      = (schedule_row.scheduled_pitcher_id == result.pitcher_id)
        game_start.was_recommended  = matched_recommendation

    # ── Update Schedule row ────────────────────────────────────
    schedule_row.actual_pitcher_id = result.pitcher_id
    schedule_row.home_score        = result.home_score
    schedule_row.away_score        = result.away_score
    schedule_row.game_completed    = True

    db.commit()

    bucket = discretize_fatigue(new_state)

    return GameResultResponse(
        game_id    = game_id,
        pitcher_id = result.pitcher_id,
        updated_fatigue = {
            "days_since_last": new_state.days_since_last,
            "rest_score":      round(new_state.rest_score,     4),
            "workload_score":  round(new_state.workload_score, 4),
            "recovery_score":  round(new_state.recovery_score, 4),
            "fatigue_bucket":  bucket,
        },
        matched_recommendation = matched_recommendation,
    )