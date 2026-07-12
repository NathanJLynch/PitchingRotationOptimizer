import pandas as pd

# database/models.py
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Float,
    ForeignKey, Index, Integer, String, Text, UniqueConstraint,
    CheckConstraint, event
)
from sqlalchemy.orm import declarative_base, relationship, validates
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from sqlalchemy.orm import DeclarativeBase
from contextlib import contextmanager
class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────
# MIXINS
# ─────────────────────────────────────────────────────────────

class TimestampMixin:
    """Adds created_at / updated_at to every model automatically."""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow, nullable=False)


class StatcastMixin:
    """
    Columns shared by both Pitcher and Team that come directly
    from Statcast pitch-level aggregations.
    """
    # Plate discipline
    zone_pct       : Mapped[float | None] = mapped_column(Float, nullable=True)   # % pitches in zone
    chase_pct      : Mapped[float | None] = mapped_column(Float, nullable=True)   # O-swing %
    whiff_pct      : Mapped[float | None] = mapped_column(Float, nullable=True)   # swinging strike %
    contact_pct    : Mapped[float | None] = mapped_column(Float, nullable=True)   # contact % on swings

    # Batted ball
    hard_hit_pct   : Mapped[float | None] = mapped_column(Float, nullable=True)   # exit velo >= 95 mph
    barrel_pct     : Mapped[float | None] = mapped_column(Float, nullable=True)   # barrels / PA
    gb_pct         : Mapped[float | None] = mapped_column(Float, nullable=True)   # ground ball %
    fb_pct         : Mapped[float | None] = mapped_column(Float, nullable=True)   # fly ball %
    ld_pct         : Mapped[float | None] = mapped_column(Float, nullable=True)   # line drive %

    # Expected stats (Statcast model outputs)
    xwoba          : Mapped[float | None] = mapped_column(Float, nullable=True)
    xslg           : Mapped[float | None] = mapped_column(Float, nullable=True)
    xba            : Mapped[float | None] = mapped_column(Float, nullable=True)


# ─────────────────────────────────────────────────────────────
# DIVISION / LEAGUE LOOKUP
# ─────────────────────────────────────────────────────────────

class Division(Base, TimestampMixin):
    """
    Static reference table: 6 MLB divisions.
    Populated once at DB init, never changes.
    """
    __tablename__ = "divisions"

    id           = Column(Integer, primary_key=True)
    name         = Column(String(20), unique=True, nullable=False)  # e.g. "NL_CENTRAL"
    league       = Column(String(2),  nullable=False)               # "AL" or "NL"
    display_name = Column(String(30), nullable=False)               # "NL Central"

    teams        = relationship("Team", back_populates="division_ref")

    def __repr__(self):
        return f"<Division {self.name}>"


# ─────────────────────────────────────────────────────────────
# TEAM
# ─────────────────────────────────────────────────────────────

class Team(Base, TimestampMixin, StatcastMixin):
    """
    One row per MLB team. Offensive profile updated daily
    by the data pipeline from Baseball Reference / Statcast.

    Relationships:
      home_games / away_games  → Schedule
      standings_snapshots      → StandingsSnapshot
      pitchers                 → Pitcher (via roster)
    """
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_team_id: Mapped[int] = mapped_column(unique=True, nullable=False)  # MLB Stats API team ID
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    abbreviation: Mapped[str] = mapped_column(String(5), nullable=False)
    division_id: Mapped[int] = mapped_column(ForeignKey("divisions.id"), nullable=False)

    # ── Offensive profile (season rolling) ───────────────────
    ops: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    woba: Mapped[float | None] = mapped_column(Float, nullable=True)
    iso: Mapped[float | None] = mapped_column(Float, nullable=True)   # isolated power
    babip: Mapped[float | None] = mapped_column(Float, nullable=True)
    k_pct: Mapped[float | None] = mapped_column(Float, nullable=True)   # team strikeout rate
    bb_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Lineup handedness composition
    rhb_pct: Mapped[float | None] = mapped_column(Float, nullable=True)   # % PA from RHB
    lhb_pct: Mapped[float | None] = mapped_column(Float, nullable=True)   # % PA from LHB

    # Per-pitch-type OPS (from Statcast)
    ops_vs_fastball: Mapped[float | None] = mapped_column(Float, nullable=True)
    ops_vs_breaking: Mapped[float | None] = mapped_column(Float, nullable=True)
    ops_vs_offspeed: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Rolling windows: separate 60-day versions for recent form
    ops_60d         : Mapped[float | None] = mapped_column(Float, nullable=True)
    woba_60d        : Mapped[float | None] = mapped_column(Float, nullable=True)
    whiff_rate_60d  : Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Metadata ──────────────────────────────────────────────
    stats_as_of     : Mapped[Date | None] = mapped_column(Date, nullable=True)   # when stats were last refreshed

    # ── Relationships ─────────────────────────────────────────
    division_ref         = relationship("Division", back_populates="teams")
    home_games           = relationship("Schedule", foreign_keys="Schedule.home_team_id",
                                        back_populates="home_team")
    away_games           = relationship("Schedule", foreign_keys="Schedule.away_team_id",
                                        back_populates="away_team")
    standings_snapshots  = relationship("StandingsSnapshot", back_populates="team")
    roster: Mapped[list["Pitcher"]] = relationship(
    back_populates="team"
)

    __table_args__ = (
        CheckConstraint("ops BETWEEN 0.3 AND 1.5",   name="ck_team_ops_range"),
        CheckConstraint("rhb_pct + lhb_pct <= 1.01", name="ck_team_handedness_sum"),
    )

    def __repr__(self):
        return f"<Team {self.abbreviation}>"


# ─────────────────────────────────────────────────────────────
# PITCHER
# ─────────────────────────────────────────────────────────────

class Pitcher(Base, TimestampMixin, StatcastMixin):
    """
    One row per pitcher on an active 40-man roster.
    Pitching profile updated daily. Fatigue state is computed
    at runtime from GameStart rows — not stored here.

    The matchup_model.py PitcherProfile dataclass is hydrated
    from this row + recent GameStart rows.
    """
    __tablename__ = "pitchers"

    id                  = Column(Integer, primary_key=True)
    mlb_player_id       = Column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    team_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("teams.id"), nullable=True)
    throws: Mapped[str] = mapped_column(String(1), nullable=False)   # "R" or "L"
    is_starter: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # ── Stuff metrics (Statcast pitch-level aggregates) ───────
    fb_velo              : Mapped[float | None] = mapped_column(Float, nullable=True)   # avg fastball velo
    fb_velo_max         : Mapped[float | None] = mapped_column(Float, nullable=True)
    spin_rate           : Mapped[float | None] = mapped_column(Float, nullable=True)   # avg spin RPM
    extension           : Mapped[float | None] = mapped_column(Float, nullable=True)   # release extension ft
    ivb                 : Mapped[float | None] = mapped_column(Float, nullable=True)   # induced vertical break
    hb                  : Mapped[float | None] = mapped_column(Float, nullable=True)   # horizontal break

    # ── Arsenal usage (must sum to ~1.0) ──────────────────────
    fastball_usage      : Mapped[float | None] = mapped_column(Float, nullable=True)
    breaking_usage      : Mapped[float | None] = mapped_column(Float, nullable=True)
    offspeed_usage      : Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Results (season + rolling) ────────────────────────────
    era                 : Mapped[float | None] = mapped_column(Float, nullable=True)
    fip                 : Mapped[float | None] = mapped_column(Float, nullable=True)
    xfip                : Mapped[float | None] = mapped_column(Float, nullable=True)
    k_pct               : Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_pct              : Mapped[float | None] = mapped_column(Float, nullable=True)
    ops_allowed         : Mapped[float | None] = mapped_column(Float, nullable=True)
    woba_allowed        : Mapped[float | None] = mapped_column(Float, nullable=True)


    # 60-day rolling (recent form — weighted more in matchup model)
    k_pct_60d           : Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_pct_60d          : Mapped[float | None] = mapped_column(Float, nullable=True)
    ops_allowed_60d     : Mapped[float | None] = mapped_column(Float, nullable=True)
    xfip_60d            : Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Platoon splits ────────────────────────────────────────
    ops_allowed_vs_rhb  : Mapped[float | None] = mapped_column(Float, nullable=True)
    ops_allowed_vs_lhb  : Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Metadata ──────────────────────────────────────────────
    stats_as_of         : Mapped[Date | None] = mapped_column(Date, nullable=True)

    # ── Relationships ─────────────────────────────────────────
    team: Mapped["Team"] = relationship(
    back_populates="roster"
)

    starts: Mapped[list["GameStart"]] = relationship(
    back_populates="pitcher",
    order_by="GameStart.game_date.desc()",
)
    __table_args__ = (
        CheckConstraint("throws IN ('R', 'L')", name="ck_pitcher_throws"),
        CheckConstraint("era BETWEEN 0 AND 15",  name="ck_pitcher_era_range"),
        Index("ix_pitchers_team_active", "team_id", "is_active", "is_starter"),
    )

    @validates("fastball_usage", "breaking_usage", "offspeed_usage")
    def validate_usage(self, key, value):
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"{key} must be between 0 and 1, got {value}")
        return value

    def __repr__(self):
        return f"<Pitcher {self.name} ({self.throws}HP)>"


# ─────────────────────────────────────────────────────────────
# SCHEDULE
# ─────────────────────────────────────────────────────────────

class Schedule(Base, TimestampMixin):
    """
    One row per scheduled game.
    The optimizer queries this to build its list of Game objects.

    scheduled_pitcher_id: what the optimizer recommended.
    actual_pitcher_id:    who actually started (set post-game).
    """
    __tablename__ = "schedule"

    id                    = Column(Integer, primary_key=True)
    mlb_game_id           = Column(BigInteger, unique=True, nullable=False)
    game_date             = Column(Date, nullable=False)
    home_team_id          = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id          = Column(Integer, ForeignKey("teams.id"), nullable=False)
    is_home               = Column(Boolean, nullable=False)   # from OUR team's perspective
    series_game_num       = Column(Integer, nullable=True)    # 1, 2, 3 in series
    h2h_remaining         = Column(Integer, nullable=True)    # h2h games left vs. opponent

    # Optimizer output
    scheduled_pitcher_id  = Column(Integer, ForeignKey("pitchers.id"), nullable=True)
    actual_pitcher_id     = Column(Integer, ForeignKey("pitchers.id"), nullable=True)
    optimizer_run_id      = Column(Integer, ForeignKey("optimizer_runs.id"), nullable=True)
    optimizer_score       = Column(Float, nullable=True)
    score_breakdown       = Column(JSON, nullable=True)     # full audit trail from dp_engine

    # Post-game result
    home_score            = Column(Integer, nullable=True)
    away_score            = Column(Integer, nullable=True)
    game_completed        = Column(Boolean, default=False)

    # ── Relationships ─────────────────────────────────────────
    home_team           = relationship("Team", foreign_keys=[home_team_id],
                                       back_populates="home_games")
    away_team           = relationship("Team", foreign_keys=[away_team_id],
                                       back_populates="away_games")
    scheduled_pitcher   = relationship("Pitcher", foreign_keys=[scheduled_pitcher_id])
    actual_pitcher      = relationship("Pitcher", foreign_keys=[actual_pitcher_id])
    optimizer_run       = relationship("OptimizerRun", back_populates="assignments")
    game_start          = relationship("GameStart", back_populates="schedule",
                                       uselist=False)

    __table_args__ = (
        CheckConstraint("home_team_id != away_team_id", name="ck_schedule_teams_differ"),
        CheckConstraint("series_game_num BETWEEN 1 AND 5", name="ck_schedule_series_num"),
        Index("ix_schedule_date_teams", "game_date", "home_team_id", "away_team_id"),
        Index("ix_schedule_date_pitcher", "game_date", "scheduled_pitcher_id"),
    )

    def __repr__(self):
        return f"<Schedule {self.game_date} game_id={self.mlb_game_id}>"


# ─────────────────────────────────────────────────────────────
# GAME START
#
# One row per actual pitching start (post-game).
# The fatigue_model.py reads this to compute FatigueState.
# Written by update_fatigue() in fatigue_model.py after each game.
# ─────────────────────────────────────────────────────────────

class GameStart(Base, TimestampMixin):
    __tablename__ = "game_starts"

    id               = Column(Integer, primary_key=True)
    pitcher_id       = Column(Integer, ForeignKey("pitchers.id"), nullable=False)
    schedule_id      = Column(Integer, ForeignKey("schedule.id"), nullable=True)
    game_date        = Column(Date, nullable=False)
    opponent_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    # Workload (what fatigue_model.py uses)
    pitch_count      = Column(Integer, nullable=False)
    innings_pitched  = Column(Float,   nullable=False)
    game_score       = Column(Integer, nullable=True)   # Bill James game score

    # Statcast performance
    velocity_delta   = Column(Float, nullable=True)    # velo vs. season avg (neg = tired)
    k_pct_actual     = Column(Float, nullable=True)    # actual K% in this start
    ops_allowed_actual = Column(Float, nullable=True)  # actual OPS allowed
    wpa              = Column(Float, nullable=True)    # win probability added

    # Was this the optimizer's recommendation?
    was_recommended  = Column(Boolean, nullable=True)

    # ── Relationships ─────────────────────────────────────────
    pitcher  = relationship("Pitcher", back_populates="starts")
    schedule = relationship("Schedule", back_populates="game_start")
    opponent = relationship("Team", foreign_keys=[opponent_team_id])

    __table_args__ = (
        UniqueConstraint("pitcher_id", "game_date", name="uq_gamestart_pitcher_date"),
        CheckConstraint("pitch_count BETWEEN 1 AND 150", name="ck_gs_pitch_count"),
        CheckConstraint("innings_pitched >= 0",           name="ck_gs_innings"),
        Index("ix_game_starts_pitcher_date", "pitcher_id", "game_date"),
    )

    def __repr__(self):
        return f"<GameStart pitcher={self.pitcher_id} date={self.game_date} pc={self.pitch_count}>"


# ─────────────────────────────────────────────────────────────
# STANDINGS SNAPSHOT
#
# One row per team per day — written nightly by data pipeline.
# division_bonus.py's _load_standings() queries this table.
# Storing daily snapshots (rather than computing live) means
# the optimizer can reason about any past or future date
# without re-fetching standings from external APIs mid-run.
# ─────────────────────────────────────────────────────────────

class StandingsSnapshot(Base, TimestampMixin):
    __tablename__ = "standings_snapshots"

    id               = Column(Integer, primary_key=True)
    team_id          = Column(Integer, ForeignKey("teams.id"), nullable=False)
    snapshot_date    = Column(Date, nullable=False)
    season           = Column(Integer, nullable=False)
    division         = Column(String(20), nullable=False)   # e.g. "NL_CENTRAL"

    # Standings position
    wins             = Column(Integer, nullable=False)
    losses           = Column(Integer, nullable=False)
    games_behind     = Column(Float, nullable=False)        # 0.0 if leading
    win_pct          = Column(Float, nullable=False)
    run_differential = Column(Integer, nullable=True)

    # Streak / form
    last_10          = Column(String(5), nullable=True)     # e.g. "7-3"
    home_record      = Column(String(10), nullable=True)    # e.g. "28-18"
    away_record      = Column(String(10), nullable=True)

    # Current offensive strength (copied from teams row on snapshot day)
    # Denormalized here so division_bonus.py gets one clean join-free query
    ops              = Column(Float, nullable=True)
    team_name        = Column(String(50), nullable=False)

    # ── Relationships ─────────────────────────────────────────
    team = relationship("Team", back_populates="standings_snapshots")

    __table_args__ = (
        UniqueConstraint("team_id", "snapshot_date", name="uq_standings_team_date"),
        CheckConstraint("wins >= 0 AND losses >= 0", name="ck_standings_record"),
        CheckConstraint("games_behind >= 0",         name="ck_standings_gb"),
        Index("ix_standings_date_division", "snapshot_date", "division"),
        Index("ix_standings_team_date",     "team_id", "snapshot_date"),
    )

    def __repr__(self):
        return (f"<StandingsSnapshot {self.team_name} "
                f"{self.snapshot_date} {self.wins}-{self.losses} "
                f"GB={self.games_behind}>")


# ─────────────────────────────────────────────────────────────
# OPTIMIZER RUN
#
# Audit log of every time dp_engine.solve() is called.
# Stores the full assignment JSON so you can replay or
# compare runs. Referenced by Schedule.optimizer_run_id.
# ─────────────────────────────────────────────────────────────

class OptimizerRun(Base, TimestampMixin):
    __tablename__ = "optimizer_runs"

    id              = Column(Integer, primary_key=True)
    our_team_id     = Column(Integer, ForeignKey("teams.id"), nullable=False)
    run_date        = Column(Date, nullable=False)
    horizon_start   = Column(Date, nullable=False)
    horizon_end     = Column(Date, nullable=False)
    horizon_days    = Column(Integer, nullable=False)

    # DP outputs
    total_score     = Column(Float, nullable=True)
    assignments     = relationship("Schedule", back_populates="optimizer_run")

    # Full serialized DP result for debugging
    assignments_json = Column(JSON, nullable=True)   # list of AssignedStart dicts
    dp_metadata      = Column(JSON, nullable=True)   # n_games, n_pitchers, FATIGUE_LEVELS

    # What triggered this run
    trigger          = Column(String(30), nullable=True)   # "scheduled" | "injury" | "manual"
    notes            = Column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint("horizon_end >= horizon_start", name="ck_run_horizon_order"),
        Index("ix_optimizer_runs_team_date", "our_team_id", "run_date"),
    )

    def __repr__(self):
        return (f"<OptimizerRun team={self.our_team_id} "
                f"date={self.run_date} score={self.total_score:.3f}>")


# ─────────────────────────────────────────────────────────────
# ML MODEL REGISTRY
#
# Tracks which model version was used in each optimizer run.
# Critical for debugging: if the optimizer starts making bad
# assignments, you need to know which model artifact to blame.
# ─────────────────────────────────────────────────────────────

class MLModelVersion(Base, TimestampMixin):
    __tablename__ = "ml_model_versions"

    id              = Column(Integer, primary_key=True)
    model_name      = Column(String(50), nullable=False)    # "k_pct_model" | "ops_model" | "wpa_model"
    version         = Column(String(20), nullable=False)    # e.g. "2025-04-01"
    artifact_path   = Column(String(255), nullable=False)   # path to .pkl file
    training_cutoff = Column(Date, nullable=False)          # last date in training data
    train_mae       = Column(Float, nullable=True)          # training MAE
    val_mae         = Column(Float, nullable=True)          # validation MAE
    feature_list    = Column(JSON, nullable=True)          # ordered feature names
    hyperparameters = Column(JSON, nullable=True)
    is_active       = Column(Boolean, default=False)        # only one active per model_name

    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_mlmodel_name_version"),
        Index("ix_mlmodel_name_active", "model_name", "is_active"),
    )

    def __repr__(self):
        return f"<MLModelVersion {self.model_name} v{self.version} active={self.is_active}>"


# ─────────────────────────────────────────────────────────────
# DB SESSION FACTORY
# ─────────────────────────────────────────────────────────────

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import os

def get_engine(database_url: str):
    return create_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,      # reconnect on stale connections
        echo=False,
    )

def get_session_factory(engine):
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)

@contextmanager
def get_db_session(session_factory):
    """
    Context manager for safe session lifecycle.
    Use everywhere outside of FastAPI dependency injection:

        with get_db_session(session_factory) as db:
            pitchers = db.query(Pitcher).filter_by(is_active=True).all()
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(database_url: str):
    """
    Creates all tables. Called once at startup or in migrations.
    Use Alembic for production schema migrations — this is
    for initial setup and testing only.
    """
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine

DATABASE_URL    = os.environ.get("DATABASE_URL", "sqlite:///rotation_optimizer.db")
engine          = get_engine(DATABASE_URL)
SessionFactory  = get_session_factory(engine)