# main.py
import asyncio
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from database.models import get_engine, get_session_factory, init_db
from api import pitchers, games
from database.pipeline.sync import sync_all


# async def _nightly_sync_loop():
#     """Runs once at midnight every day."""
#     while True:
#         now = __import__("datetime").datetime.now()
#         # Sleep until next midnight
#         tomorrow = (now + __import__("datetime").timedelta(days=1)).replace(
#             hour=0, minute=5, second=0, microsecond=0
#         )
#         seconds_until_midnight = (tomorrow - now).total_seconds()
#         await asyncio.sleep(seconds_until_midnight)

#         try:
#             logger.info("Running nightly sync...")
#             run_sync(app.state.session_factory)
#             logger.info("Nightly sync complete.")
#         except Exception as e:
#             logger.error(f"Nightly sync failed: {e}")

# asyncio.create_task(_nightly_sync_loop())

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rotation_optimizer")


# ─────────────────────────────────────────────────────────────
# CONFIG
#
# Pulled from environment. In production these come from
# whatever secrets manager / .env loader you use.
# ─────────────────────────────────────────────────────────────

class Settings:
    DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "sqlite:///rotation_optimizer.db"
    )
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development")
    CORS_ORIGINS: list = os.environ.get("CORS_ORIGINS", "*").split(",")
    AUTO_CREATE_TABLES: bool = os.environ.get("AUTO_CREATE_TABLES", "false").lower() == "true"
    ML_MODEL_DIR: str = os.environ.get("ML_MODEL_DIR", "ml/models")


settings = Settings()


# ─────────────────────────────────────────────────────────────
# DB SESSION FACTORY — APP STATE
#
# Created once at startup, stored on app.state.
# Injected into route dependencies via dependency_overrides.
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: create engine + session factory, optionally init schema,
    verify ML models are loadable.

    Shutdown: dispose engine connections cleanly.
    """
    logger.info(f"Starting rotation optimizer API [{settings.ENVIRONMENT}]")

    # ── Database setup ────────────────────────────────────────
    engine = get_engine(settings.DATABASE_URL)

    if settings.AUTO_CREATE_TABLES:
        logger.warning("AUTO_CREATE_TABLES=true — creating tables if missing. "
                       "Use Alembic migrations in production instead.")
        init_db(settings.DATABASE_URL)

    session_factory = get_session_factory(engine)
    app.state.engine = engine
    app.state.session_factory = session_factory

    async def _daily_sync():
        while True:
            try:
                sync_all(app.state.session_factory, None)
            except Exception as e:
                logger.error(f"Pipeline sync failed: {e}")
            await asyncio.sleep(60 * 60 * 24)   # 24 hours

    asyncio.create_task(_daily_sync())

    # ── Wire dependency overrides for both routers ────────────
    _wire_db_dependency(app, pitchers, session_factory)
    _wire_db_dependency(app, games, session_factory)

    # ── Verify ML models load correctly at startup ─────────────
    # Fail fast if model artifacts are missing rather than erroring
    # on the first optimizer request.
    try:
        from ml.predict import PitcherScoringPredictor
        _ = PitcherScoringPredictor()
        logger.info("ML models loaded successfully")
    except FileNotFoundError as e:
        logger.error(
            f"ML model artifacts not found: {e}. "
            f"Run `python ml/train.py` before starting the API, "
            f"or set ML_MODEL_DIR to point at existing .pkl files."
        )
        if settings.ENVIRONMENT == "production":
            raise

    logger.info("Startup complete")

    yield

    # ── Shutdown ────────────────────────────────────────────────
    logger.info("Shutting down — disposing DB engine")
    engine.dispose()


def _wire_db_dependency(app: FastAPI, module, session_factory):
    """
    Each router module defines:

        def get_db(session_factory=Depends(lambda: None)):
            with get_db_session(session_factory) as db:
                yield db

    This override replaces the inner lambda dependency with one
    that returns the real session_factory, so get_db() yields a
    working session.
    """
    def _session_factory_dependency():
        return session_factory

    # Find the placeholder lambda used as the default for session_factory
    # FastAPI resolves Depends() by identity, so we override at the
    # get_db function's own dependency signature.
    app.dependency_overrides[module.get_db] = _make_get_db_override(module, session_factory)


def _make_get_db_override(module, session_factory):
    """
    Returns a generator function matching get_db()'s signature,
    but bound to the real session_factory.
    """
    from database.models import get_db_session

    def _override():
        with get_db_session(session_factory) as db:
            yield db

    return _override


# ─────────────────────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Pitching Rotation Optimizer",
    description=(
        "DP-based starting pitcher optimizer. Combines matchup analysis "
        "(whiff rates, OPS suppression, platoon splits), fatigue/workload "
        "modeling, division-race urgency, and ML-derived scoring coefficients "
        "to recommend optimal starter assignments over a rolling horizon."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────────────────────

app.include_router(pitchers.router)
app.include_router(games.router)


# ─────────────────────────────────────────────────────────────
# GLOBAL ERROR HANDLING
# ─────────────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """
    Catches ValueErrors raised from dp_engine, fatigue_model, etc.
    that weren't already wrapped as HTTPExceptions.
    """
    logger.error(f"ValueError on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catch-all. In production, never leak stack traces to the client —
    log full detail server-side, return a generic message.
    """
    logger.exception(f"Unhandled exception on {request.url.path}")

    if settings.ENVIRONMENT == "development":
        detail = f"{type(exc).__name__}: {exc}"
    else:
        detail = "Internal server error"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": detail},
    )


# ─────────────────────────────────────────────────────────────
# HEALTH / META ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health_check():
    """Basic liveness check — does not touch the DB."""
    return {"status": "ok", "environment": settings.ENVIRONMENT}


@app.get("/health/db", tags=["meta"])
def db_health_check():
    """
    Verifies DB connectivity. Used by deployment readiness probes.
    """
    from sqlalchemy import text

    try:
        with app.state.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "database": str(e)},
        )


@app.get("/health/models", tags=["meta"])
def ml_model_health_check():
    """
    Verifies the active ML model versions are loadable and
    reports their training metadata.
    """
    from database.models import MLModelVersion, get_db_session

    try:
        with get_db_session(app.state.session_factory) as db:
            active_models = (
                db.query(MLModelVersion)
                .filter_by(is_active=True)
                .all()
            )

        if not active_models:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "error", "detail": "No active ML models registered"},
            )

        return {
            "status": "ok",
            "active_models": [
                {
                    "model_name":      m.model_name,
                    "version":         m.version,
                    "training_cutoff": m.training_cutoff.isoformat(),
                    "val_mae":         m.val_mae,
                }
                for m in active_models
            ],
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "detail": str(e)},
        )


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Pitching Rotation Optimizer",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "rotation":           "/pitchers/team/{team_id}/rotation",
            "fatigue":            "/pitchers/{pitcher_id}/fatigue",
            "matchup_preview":    "/pitchers/{pitcher_id}/matchup/{opponent_team_id}",
            "upcoming_games":     "/games/upcoming?team_id={id}",
            "run_optimizer":      "POST /games/optimizer/run",
            "latest_assignments": "/games/optimizer/assignments?team_id={id}",
            "record_result":      "POST /games/{game_id}/result",
        },
    }


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=(settings.ENVIRONMENT == "development"),
        log_level="info",
    )