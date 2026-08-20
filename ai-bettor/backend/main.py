"""Main FastAPI application for AI Bettor."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database.models import (
    AgentAnalysis, BankrollRecord, Bet, Match, OddsSnapshot,
    Prediction, RiskAssessment, Simulation, SystemLog,
)
from backend.database.session import database_info, get_db, init_db
from backend.security import TokenAuthMiddleware, describe_auth, parse_origins, token_matches
from backend.services.backtest import Backtester, get_backtester
from backend.services.paper_betting import PaperBettingService, get_paper_betting
from backend.services.pipeline import AiBettorPipeline, get_pipeline
from backend.services.scoring import PickScoringEngine, get_scoring_engine
from backend.services.settings_service import get_revision, get_setting

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ai-bettor")

settings = get_settings()


def _live(key: str, default: Any = None) -> Any:
    """One live setting, falling back to the .env snapshot then `default`.

    Every endpoint reads through this so a value saved from the dashboard is
    reflected immediately, without restarting the process.
    """
    value = get_setting(key, getattr(settings, key, default))
    return default if value is None else value


def _live_int(key: str, default: int) -> int:
    try:
        return int(_live(key, default))
    except (TypeError, ValueError):
        return default


def _live_bool(key: str, default: bool) -> bool:
    value = _live(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown. Boots the DB then the automation scheduler."""
    logger.info("AI Bettor starting up...")
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database init failed: %s", e)

    logger.info("Timezone: %s", _live("TIMEZONE", "Asia/Jakarta"))
    logger.info("Monte Carlo: %s x%s batches",
                _live_int("MONTE_CARLO_SIMULATIONS", 20000),
                _live_int("SIMULATION_BATCHES", 3))
    logger.info("Betting mode: %s", _live("BETTING_MODE", "PAPER"))
    logger.info("Early-morning window (dini hari WIB): 00:00 - %02d:00",
                _live_int("EARLY_MORNING_END_HOUR", 6))
    logger.info("Telegram min score: %s, max picks: %s",
                _live_int("TELEGRAM_MIN_SCORE", 85),
                _live_int("TELEGRAM_MAX_PICKS", 5))

    if _live_bool("SCAN_AUTOMATION_ENABLED", True):
        _automation["task"] = asyncio.create_task(_automation_loop())
        logger.info("Automation scheduler started")
    logger.info("AI Bettor ready.")

    try:
        yield
    finally:
        # Cancel whatever task is current — /automation/toggle may have
        # replaced the one we started here.
        task = _automation.get("task")
        _automation["task"] = None
        if task is not None:
            task.cancel()
            try:
                await task
            except BaseException:
                pass


app = FastAPI(
    title="AI Bettor",
    description="Autonomous AI sports betting analysis agent",
    version="0.1.0",
    lifespan=lifespan,
)

# --- access control ------------------------------------------------
# Middleware is applied outside-in in reverse order of registration, so CORS is
# added last and therefore wraps the auth guard: a rejected request still comes
# back with CORS headers, which is what lets the dashboard read the 401 instead
# of reporting an opaque network error.
_api_token = (settings.API_TOKEN or "").strip()
_allowed_origins = parse_origins(settings.ALLOWED_ORIGINS)

app.add_middleware(TokenAuthMiddleware, token=_api_token)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    # Credentials are pointless here (the token travels in a header, not a
    # cookie) and cannot be combined with a wildcard origin, so they stay off.
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Token"],
)

if not _api_token:
    logger.warning(
        "API_TOKEN is not set — every endpoint is open. Fine for localhost, "
        "not for a public VPS: set API_TOKEN in .env before exposing port 8000."
    )
if _allowed_origins == ["*"]:
    logger.warning(
        "ALLOWED_ORIGINS is not set — CORS accepts any origin. Set it to your "
        "dashboard URL (e.g. https://bettor.example.com) in production."
    )

# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    database: str
    # Enough for the dashboard and a deploy smoke test to confirm *which*
    # database and whether the API is guarded. Never any secret value.
    database_info: Dict[str, Any] = Field(default_factory=dict)
    auth: Dict[str, Any] = Field(default_factory=dict)


class ScanRequest(BaseModel):
    # All optional: omitted fields fall back to the live settings, so the
    # dashboard never has to restate the saved scan configuration. The old
    # defaults ("idf" regions, a "1X2" market) are not valid Odds API values.
    sport: Optional[str] = None
    regions: Optional[str] = None
    markets: Optional[str] = None
    odds_format: str = "decimal"
    early_morning_only: Optional[bool] = None


class AnalyzeRequest(BaseModel):
    match_id: str
    model_probability: float = Field(gt=0, lt=1)
    decimal_odds: float = Field(gt=1)
    bookmaker_margin: Optional[float] = None


class SimulateRequest(BaseModel):
    home_lambda: float = Field(gt=0)
    away_lambda: float = Field(gt=0)
    simulation_count: Optional[int] = Field(default=None, ge=100)
    random_seed: Optional[int] = None


class RiskRequest(BaseModel):
    quant_probability: float = Field(gt=0, lt=1)
    decimal_odds: float = Field(gt=1)
    current_exposure: float = 0.0
    current_drawdown: float = 0.0
    bankroll: float = 1000.0
    recent_results: Optional[List[str]] = None


class DecideRequest(BaseModel):
    quant_result: Dict[str, Any]
    market_result: Dict[str, Any] = {}
    simulation_result: Dict[str, Any]
    risk_result: Dict[str, Any] = {}
    match_data: Dict[str, Any] = {}
    bettor_state: Dict[str, Any] = {}


class BacktestRequest(BaseModel):
    matches: List[Dict[str, Any]]


# ------------------------------------------------------------------
# Lazy singletons (rebuilt whenever the settings revision moves)
# ------------------------------------------------------------------

_paper: Optional[PaperBettingService] = None
_scoring: Optional[PickScoringEngine] = None
_cached_revision: int = -1
_singleton_lock = threading.Lock()


def _get_pipeline() -> AiBettorPipeline:
    """The shared pipeline. `get_pipeline()` caches it and refreshes it when
    settings change, so we must not hold a stale module-level copy here."""
    return get_pipeline()


def _invalidate_if_stale() -> None:
    global _paper, _scoring, _cached_revision
    revision = get_revision()
    if revision != _cached_revision:
        with _singleton_lock:
            _paper = None
            _scoring = None
            _cached_revision = revision


def _get_paper() -> PaperBettingService:
    global _paper
    _invalidate_if_stale()
    if _paper is None:
        _paper = get_paper_betting()
    return _paper


def _get_scoring() -> PickScoringEngine:
    global _scoring
    _invalidate_if_stale()
    if _scoring is None:
        _scoring = get_scoring_engine()
    return _scoring


# ------------------------------------------------------------------
# System endpoints
# ------------------------------------------------------------------

@app.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check(db: Session = Depends(get_db)) -> HealthCheck:
    from backend.integrations.odds_router import get_odds_router

    router = get_odds_router()
    services: Dict[str, str] = {}
    services["the_odds_api"] = "configured" if router.has_keys else "missing_key"
    services["openrouter"] = "configured" if _live("OPENROUTER_API_KEY", "") else "missing_key"
    services["telegram"] = (
        "configured"
        if (_live("TELEGRAM_BOT_TOKEN", "") and _live("TELEGRAM_CHAT_ID", ""))
        else "missing_key"
    )

    db_status = "connected"
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_status = "error"

    status = "ok"
    if "missing_key" in services.values() or db_status == "error":
        status = "degraded"

    return HealthCheck(
        status=status,
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        services=services,
        database=db_status,
        database_info=database_info(),
        auth=describe_auth(_api_token),
    )


@app.get("/auth/check", tags=["System"])
async def auth_check(request: Request):
    """Does the token this request carries work?

    Public so the dashboard can validate a freshly pasted token and show a clear
    message, rather than every panel failing with a bare 401.
    """
    if not _api_token:
        return {"auth_required": False, "valid": True}
    supplied = request.headers.get("X-API-Token") or ""
    if not supplied:
        authorization = request.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
    return {"auth_required": True, "valid": token_matches(supplied.strip(), _api_token)}


@app.get("/metrics", tags=["System"])
async def metrics():
    return {
        "uptime_seconds": round(time.time() - _start_time),
        "agents": len(_get_pipeline().agent_status()),
        "mode": _live("BETTING_MODE", "PAPER"),
        "timezone": _live("TIMEZONE", "Asia/Jakarta"),
        "monte_carlo_simulations": _live_int("MONTE_CARLO_SIMULATIONS", 20000),
        "simulation_batches": _live_int("SIMULATION_BATCHES", 3),
        "random_seed": _live_int("RANDOM_SEED", 42),
        "scan_interval_seconds": _live_int("AGENT_SCAN_INTERVAL_SECONDS", 900),
        "odds_poll_interval_seconds": getattr(settings, "ODDS_POLL_INTERVAL_SECONDS", None),
        "score_bet_threshold": _live_int("SCORE_BET_THRESHOLD", 80),
        "settings_revision": get_revision(),
    }


@app.get("/logs", tags=["System"])
async def get_logs(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    rows = db.query(SystemLog).order_by(SystemLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "log_id": r.log_id, "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "service": r.service, "agent": r.agent, "match_id": r.match_id,
            "action": r.action, "status": r.status, "latency": r.latency,
            "error_details": r.error_details,
        }
        for r in rows
    ]


# ------------------------------------------------------------------
# Match / odds endpoints
# ------------------------------------------------------------------

@app.get("/matches", tags=["Matches"])
async def list_matches(
    league: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    q = db.query(Match)
    if league:
        q = q.filter(Match.league == league)
    if status:
        q = q.filter(Match.status == status)
    rows = q.order_by(Match.kickoff.asc()).limit(limit).all()
    return [
        {
            "match_id": m.match_id, "home_team": m.home_team, "away_team": m.away_team,
            "kickoff": m.kickoff.isoformat() if m.kickoff else None, "league": m.league,
            "sport": m.sport, "status": m.status,
        }
        for m in rows
    ]


@app.get("/matches/{match_id}", tags=["Matches"])
async def get_match(match_id: str, db: Session = Depends(get_db)):
    match = db.query(Match).filter(Match.match_id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    odds = db.query(OddsSnapshot).filter(OddsSnapshot.match_id == match_id).all()
    predictions = db.query(Prediction).filter(Prediction.match_id == match_id).all()
    simulations = db.query(Simulation).filter(Simulation.match_id == match_id).all()
    risks = db.query(RiskAssessment).filter(RiskAssessment.match_id == match_id).all()

    return {
        "match_id": match.match_id,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "kickoff": match.kickoff.isoformat() if match.kickoff else None,
        "league": match.league,
        "sport": match.sport,
        "status": match.status,
        "odds": [
            {
                "bookmaker": o.bookmaker, "market": o.market, "selection": o.selection,
                "line": o.line, "odds": o.odds,
                "timestamp": o.timestamp.isoformat() if o.timestamp else None,
            }
            for o in odds
        ],
        "predictions": [
            {
                "decision": p.decision, "market": p.market, "selection": p.selection,
                "odds": p.odds, "model_probability": p.model_probability,
                "edge": p.edge, "ev": p.ev, "confidence_score": p.confidence_score,
                "pick_score": p.pick_score, "score_label": p.score_label,
                "risk_level": p.risk_level,
            }
            for p in predictions
        ],
        "simulations": [
            {
                "home_win_probability": s.home_win_probability,
                "draw_probability": s.draw_probability,
                "away_win_probability": s.away_win_probability,
                "variance": s.variance, "stability": s.stability,
                "simulation_count": s.simulation_count,
            }
            for s in simulations
        ],
        "risk_assessments": [
            {"risk_level": r.risk_level, "veto_decision": r.veto_decision,
             "exposure": r.exposure, "drawdown": r.drawdown}
            for r in risks
        ],
    }


@app.get("/odds", tags=["Matches"])
async def list_odds(
    match_id: Optional[str] = None,
    market: Optional[str] = None,
    bookmaker: Optional[str] = None,
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    q = db.query(OddsSnapshot)
    if match_id:
        q = q.filter(OddsSnapshot.match_id == match_id)
    if market:
        q = q.filter(OddsSnapshot.market == market)
    if bookmaker:
        q = q.filter(OddsSnapshot.bookmaker == bookmaker)
    rows = q.order_by(OddsSnapshot.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": o.id, "match_id": o.match_id, "bookmaker": o.bookmaker,
            "market": o.market, "selection": o.selection, "line": o.line,
            "odds": o.odds, "timestamp": o.timestamp.isoformat() if o.timestamp else None,
        }
        for o in rows
    ]


# ------------------------------------------------------------------
# Prediction endpoints
# ------------------------------------------------------------------

@app.get("/predictions", tags=["Predictions"])
async def list_predictions(
    decision: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    q = db.query(Prediction)
    if decision:
        q = q.filter(Prediction.decision == decision.upper())
    rows = q.order_by(Prediction.created_at.desc()).limit(limit).all()
    return [
        {
            "prediction_id": p.prediction_id, "match_id": p.match_id,
            "decision": p.decision, "market": p.market, "selection": p.selection,
            "odds": p.odds, "bookmaker": p.bookmaker,
            "model_probability": p.model_probability,
            "implied_probability": p.implied_probability,
            "edge": p.edge, "ev": p.ev, "confidence_score": p.confidence_score,
            "pick_score": p.pick_score, "score_label": p.score_label,
            "risk_level": p.risk_level,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in rows
    ]


@app.get("/predictions/{prediction_id}", tags=["Predictions"])
async def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    p = db.query(Prediction).filter(Prediction.prediction_id == prediction_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {
        "prediction_id": p.prediction_id, "match_id": p.match_id,
        "decision": p.decision, "market": p.market, "selection": p.selection,
        "odds": p.odds, "bookmaker": p.bookmaker,
        "model_probability": p.model_probability,
        "implied_probability": p.implied_probability,
        "edge": p.edge, "ev": p.ev, "confidence_score": p.confidence_score,
        "pick_score": p.pick_score, "score_label": p.score_label,
        "risk_level": p.risk_level, "reasoning": p.reasoning,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


# ------------------------------------------------------------------
# Agent endpoints
# ------------------------------------------------------------------

@app.get("/agents", tags=["Agents"])
async def list_agents():
    return _get_pipeline().agent_status()


@app.get("/agents/status", tags=["Agents"])
async def agents_status():
    return _get_pipeline().agent_status()


# ------------------------------------------------------------------
# Analysis / simulation / risk / decision endpoints
# ------------------------------------------------------------------

@app.post("/scan", tags=["Analysis"])
async def run_scan(request: ScanRequest):
    pipeline = _get_pipeline()
    pipeline.scout.max_retries = 3
    summary = await asyncio.to_thread(
        pipeline.run_cycle,
        early_morning_only=request.early_morning_only,
        sports=request.sport,
        regions=request.regions,
        markets=request.markets,
    )
    summary["agents"] = pipeline.agent_status()
    return summary


@app.post("/analyze", tags=["Analysis"])
async def analyze_match(request: AnalyzeRequest):
    quant = _get_pipeline().quant
    result = quant.analyze(request.match_id, request.model_probability, request.decimal_odds,
                           request.bookmaker_margin)
    return result.to_dict()


@app.post("/simulate", tags=["Simulation"])
async def run_simulation(request: SimulateRequest):
    sim = _get_pipeline().simulation
    result = sim.simulate(request.home_lambda, request.away_lambda,
                          request.simulation_count, request.random_seed)
    return result.to_dict()


@app.post("/risk-assess", tags=["Risk"])
async def assess_risk(request: RiskRequest):
    from backend.agents.quant_analyst import QuantAnalystResult
    from backend.agents.simulation_analyst import SimulationAnalystResult

    quant_mock = QuantAnalystResult()
    quant_mock.model_probability = request.quant_probability
    quant_mock.edge = request.quant_probability - (1 / request.decimal_odds)
    quant_mock.ev = (request.quant_probability * (request.decimal_odds - 1)) - ((1 - request.quant_probability) * 1)
    quant_mock.confidence_score = 70
    quant_mock.risk_level = "MEDIUM"

    sim_mock = SimulationAnalystResult()
    sim_mock.home_win_probability = request.quant_probability
    sim_mock.draw_probability = 0.2
    sim_mock.away_win_probability = 0.3
    sim_mock.variance = 2.0
    sim_mock.stability = 0.85

    risk = _get_pipeline().risk
    result = risk.assess(
        quant_mock, sim_mock,
        [{"name": "Unknown", "markets": [{"key": "1X2", "selections": [{"name": "Home", "odd": request.decimal_odds}]}]}],
        current_exposure=request.current_exposure,
        current_drawdown=request.current_drawdown,
        bankroll=request.bankroll,
        recent_results=request.recent_results,
    )
    return result.to_dict()


@app.post("/decide", tags=["Decision"])
async def make_decision(request: DecideRequest):
    brain = _get_pipeline().brain
    result = brain.decide(
        request.quant_result,
        request.market_result,
        request.simulation_result,
        request.risk_result,
        request.match_data,
        request.bettor_state,
    )
    return result.to_dict()


# ------------------------------------------------------------------
# Performance / bankroll / backtest endpoints
# ------------------------------------------------------------------

@app.get("/performance", tags=["Performance"])
async def performance(db: Session = Depends(get_db)):
    bets = db.query(Bet).filter(Bet.status == "settled").all()
    total_bets = len(bets)
    wins = sum(1 for b in bets if b.result == "win")
    losses = sum(1 for b in bets if b.result == "loss")
    pushes = sum(1 for b in bets if b.result == "push")
    total_staked = sum(b.stake or 0.0 for b in bets)
    # Gross return: a winner pays stake x odds (stake included), a push returns
    # the stake, a loser returns nothing. Profit is return minus stake.
    total_returned = sum(
        (b.stake or 0.0) * (b.odds or 0.0) if b.result == "win"
        else ((b.stake or 0.0) if b.result == "push" else 0.0)
        for b in bets
    )
    profit = total_returned - total_staked
    roi = (profit / total_staked) if total_staked > 0 else 0.0

    by_market: Dict[str, Dict[str, Any]] = {}
    for b in bets:
        key = b.market or "UNKNOWN"
        entry = by_market.setdefault(key, {"bets": 0, "wins": 0, "profit": 0.0, "staked": 0.0})
        entry["bets"] += 1
        entry["wins"] += 1 if b.result == "win" else 0
        entry["staked"] += b.stake or 0.0
        entry["profit"] += (
            (b.stake or 0.0) * ((b.odds or 1.0) - 1)
            if b.result == "win"
            else (-(b.stake or 0.0) if b.result == "loss" else 0.0)
        )
    for entry in by_market.values():
        entry["profit"] = round(entry["profit"], 2)
        entry["staked"] = round(entry["staked"], 2)
        entry["roi"] = round(entry["profit"] / entry["staked"], 4) if entry["staked"] else 0.0

    by_confidence: Dict[str, Dict[str, Any]] = {}
    for p in db.query(Prediction).all():
        score = p.confidence_score or 0
        key = "90+" if score >= 90 else "80-89" if score >= 80 else "70-79" if score >= 70 else "<70"
        entry = by_confidence.setdefault(key, {"predictions": 0, "bets": 0})
        entry["predictions"] += 1
        entry["bets"] += 1 if p.decision == "BET" else 0

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / total_bets, 4) if total_bets else 0,
        "profit_loss": round(profit, 2),
        "roi": round(roi, 4),
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_returned, 2),
        "average_odds": round(sum(b.odds or 0 for b in bets) / total_bets, 4) if total_bets else 0,
        "by_market": by_market,
        "by_confidence": by_confidence,
    }


@app.get("/bankroll", tags=["Bankroll"])
async def bankroll():
    return _get_paper().get_bankroll()


@app.get("/bets", tags=["Bankroll"])
async def bets_history(limit: int = Query(50, ge=1, le=500)):
    return _get_paper().history(limit)


@app.post("/bets/settle", tags=["Bankroll"])
async def settle_bet(bet_id: str, outcome: str):
    return _get_paper().settle_bet(bet_id, outcome)


@app.post("/backtest", tags=["Backtesting"])
async def run_backtest(request: BacktestRequest):
    backtester = get_backtester()
    result = backtester.run(request.matches)
    return result.to_dict()


@app.get("/scoring/thresholds", tags=["Scoring"])
async def scoring_thresholds():
    engine = _get_scoring()
    return {
        "no_bet": engine.thresholds.no_bet_max,
        "pass": engine.thresholds.pass_max,
        "watch": engine.thresholds.watch_max,
        "bet": engine.thresholds.bet_max,
        "premium": engine.thresholds.premium_min,
        "labels": ["NO BET", "PASS", "WATCH", "BET CANDIDATE", "PREMIUM CANDIDATE"],
    }


# ------------------------------------------------------------------
# Settings endpoints (API keys, strategy params, multi-key router)
# ------------------------------------------------------------------

class SettingsPayload(BaseModel):
    """Every writable setting. Anything omitted is left untouched.

    The previous version listed only a third of the keys, so saving the rest
    from the dashboard silently did nothing — that was the "settingan tidak
    dapat disimpan" bug. `extra="allow"` lets a newly added key reach the
    service (which validates it and reports unknown keys) instead of being
    dropped by the request model.
    """

    model_config = {"extra": "allow"}

    # integrations
    THE_ODDS_API_KEYS: Optional[List[str]] = None
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: Optional[str] = None
    LLM_REVIEW_ENABLED: Optional[bool] = None
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    # scanning
    DEFAULT_SPORT: Optional[str] = None
    DEFAULT_REGIONS: Optional[str] = None
    DEFAULT_MARKETS: Optional[str] = None
    MAX_LEAGUES_PER_SCAN: Optional[int] = None
    AGENT_SCAN_INTERVAL_SECONDS: Optional[int] = None
    SCAN_AUTOMATION_ENABLED: Optional[bool] = None
    EARLY_MORNING_ONLY: Optional[bool] = None
    EARLY_MORNING_END_HOUR: Optional[int] = None
    EARLY_MORNING_DAYS: Optional[int] = None
    # modelling
    MONTE_CARLO_SIMULATIONS: Optional[int] = None
    SIMULATION_BATCHES: Optional[int] = None
    RANDOM_SEED: Optional[int] = None
    MODEL_BLEND_WEIGHT: Optional[float] = None
    DEFAULT_TOTAL_GOALS: Optional[float] = None
    # strategy
    MIN_ODDS: Optional[float] = None
    MAX_ODDS: Optional[float] = None
    MIN_EDGE: Optional[float] = None
    MIN_EV: Optional[float] = None
    MIN_CONFIDENCE: Optional[int] = None
    MIN_BOOKMAKERS: Optional[int] = None
    MIN_DATA_QUALITY: Optional[int] = None
    SCORE_BET_THRESHOLD: Optional[int] = None
    MAX_UNCERTAINTY: Optional[float] = None
    # bankroll
    BETTING_MODE: Optional[str] = None
    INITIAL_BANKROLL: Optional[float] = None
    KELLY_FRACTION: Optional[float] = None
    MAX_STAKE_PERCENT: Optional[float] = None
    MAX_EXPOSURE_PERCENT: Optional[float] = None
    # notifications
    TELEGRAM_MIN_SCORE: Optional[int] = None
    TELEGRAM_MAX_PICKS: Optional[int] = None
    TELEGRAM_SEND_NO_BET: Optional[bool] = None
    # misc
    TIMEZONE: Optional[str] = None


class TestOddsKeyRequest(BaseModel):
    """Test a pasted key, or a stored one by its index in the router."""
    api_key: Optional[str] = None
    index: Optional[int] = None


class TestOpenRouterRequest(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None


class TestTelegramRequest(BaseModel):
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    message: Optional[str] = None


_settings_service: Optional[Any] = None  # kept for backwards compatibility


def _get_settings_service():
    """Always the shared service — caching it here would strand tests (and
    `reset_settings_service()`) on a stale instance."""
    from backend.services.settings_service import get_settings_service
    return get_settings_service()


@app.get("/settings", tags=["Settings"])
async def get_settings_api():
    return _get_settings_service().masked_view()


@app.put("/settings", tags=["Settings"])
async def update_settings_api(payload: SettingsPayload):
    svc = _get_settings_service()
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        view = svc.masked_view()
        view.update({"applied": {}, "rejected": {}, "failed": [], "saved": True})
        return view
    result = svc.update(updates)
    # Rebuild everything that caches a setting so the change is live at once.
    _get_pipeline().refresh_settings()
    _invalidate_if_stale()
    return result


@app.post("/settings/reload", tags=["Settings"])
async def reload_settings_api():
    """Re-read settings from the database and rebuild the engine."""
    view = _get_settings_service().reload()
    _get_pipeline().refresh_settings()
    _invalidate_if_stale()
    return view


@app.get("/settings/odds-api/status", tags=["Settings"])
async def odds_api_router_status():
    svc = _get_settings_service()
    from backend.integrations.odds_router import mask_key
    return {
        "keys": [
            {**entry, "full_key": mask_key(entry["full_key"])}
            for entry in svc.router.status()
        ],
        "has_keys": svc.router.has_keys,
        "active_key": svc.router.active_key_label(),
    }


@app.post("/settings/test-odds-key", tags=["Settings"])
async def test_odds_key(request: TestOddsKeyRequest):
    """Test a pasted key, or a stored key by index (secrets are never echoed)."""
    from backend.agents.data_scout import test_odds_api_key

    key = request.api_key or ""
    label = "pasted key"
    if not key and request.index is not None:
        stored = _get_settings_service().get_odds_api_keys()
        if not 0 <= request.index < len(stored):
            return {"success": False, "message": f"No stored key at index {request.index}"}
        key = stored[request.index]
        label = f"stored key #{request.index}"
    result = await asyncio.to_thread(test_odds_api_key, key)
    result["tested"] = label
    return result


@app.post("/settings/test-telegram", tags=["Settings"])
async def test_telegram(request: TestTelegramRequest):
    """Send a test message with either the supplied or the stored credentials."""
    from backend.integrations.telegram import TelegramNotifier

    svc = _get_settings_service()
    token = request.bot_token or svc.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = request.chat_id or svc.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        missing = "TELEGRAM_BOT_TOKEN" if not token else "TELEGRAM_CHAT_ID"
        return {"success": False, "message": f"{missing} is not set"}

    notifier = TelegramNotifier(bot_token=token, chat_id=chat_id, timeout=15)
    text = request.message or (
        "AI Bettor test message - koneksi Telegram berhasil."
    )
    ok = await asyncio.to_thread(notifier.send_message, text)
    return {
        "success": ok,
        "message": "Test message sent" if ok else "Telegram rejected the message (check token/chat id)",
    }


@app.post("/settings/test-openrouter", tags=["Settings"])
async def test_openrouter(request: TestOpenRouterRequest):
    from backend.integrations.openrouter import OpenRouterClient

    key = request.api_key or _get_settings_service().get("OPENROUTER_API_KEY", "")
    model = request.model or _get_settings_service().get("OPENROUTER_MODEL", "openrouter/auto")
    if not key:
        return {"success": False, "message": "OpenRouter API key is empty"}
    try:
        client = OpenRouterClient(api_key=key, model=model, timeout=20, max_retries=1)
        reply = client.complete(
            system_prompt="Reply with exactly: OK",
            user_prompt="Connection test",
        )
        return {"success": True, "message": f"Connected. Model replied: {reply.strip()[:80]}", "model": model}
    except Exception as e:
        return {"success": False, "message": str(e), "model": model}


# ------------------------------------------------------------------
# Automation scheduler
# ------------------------------------------------------------------

_start_time = time.time()

# One shared record of what the scheduler is doing. `running` is the overlap
# guard: a manual trigger and the timer can never run a cycle at the same time.
_automation: Dict[str, Any] = {
    "task": None,
    "running": False,
    "last_trigger": None,
    "last_started_at": None,
    "last_finished_at": None,
    "last_summary": None,
    "last_error": None,
    "cycles_completed": 0,
    "next_run_at": None,
}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


async def _run_cycle_once(trigger: str) -> Dict[str, Any]:
    """Run exactly one pipeline cycle. Never raises, never overlaps."""
    if _automation["running"]:
        return {"status": "skipped", "reason": "cycle_already_running"}

    _automation.update({"running": True, "last_trigger": trigger,
                        "last_started_at": _now_iso()})
    try:
        summary = await asyncio.to_thread(_get_pipeline().run_cycle)
        _automation["last_summary"] = {
            k: v for k, v in summary.items() if k not in ("picks", "no_bets")
        }
        _automation["last_error"] = None
        _automation["cycles_completed"] += 1
        return summary
    except Exception as e:
        logger.error("%s scan failed: %s", trigger, e)
        _automation["last_error"] = str(e)
        return {"status": "failed", "error": str(e)}
    finally:
        _automation["running"] = False
        _automation["last_finished_at"] = _now_iso()


async def _automation_loop() -> None:
    """Background scheduler: one full scan cycle every scan interval.

    The interval and the on/off switch are re-read from live settings on every
    iteration, so changing them in the dashboard takes effect without a
    restart. Cycles are skipped (never queued) when one is still running or no
    Odds API key is configured.
    """
    logger.info("Automation loop armed")
    while True:
        try:
            interval = max(30, _live_int("AGENT_SCAN_INTERVAL_SECONDS", 900))
            _automation["next_run_at"] = (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=interval)
            ).isoformat()
            await asyncio.sleep(interval)

            if not _live_bool("SCAN_AUTOMATION_ENABLED", True):
                continue
            from backend.integrations.odds_router import get_odds_router
            if not get_odds_router().has_keys:
                logger.info("Automation: no Odds API key configured, cycle skipped")
                continue

            summary = await _run_cycle_once("automation")
            logger.info(
                "Auto-scan %s: %s matches, %s BET, %s telegram",
                summary.get("status", "?"),
                summary.get("matches_scanned", 0),
                summary.get("bet_candidates", 0),
                summary.get("telegram_sent", 0),
            )
        except asyncio.CancelledError:
            logger.info("Automation loop stopped")
            break
        except Exception as e:
            logger.warning("Automation loop error: %s", e)
            await asyncio.sleep(30)


def _automation_alive() -> bool:
    task = _automation.get("task")
    return bool(task) and not task.done()


@app.get("/automation/status", tags=["Automation"])
async def automation_status():
    return {
        "enabled": _live_bool("SCAN_AUTOMATION_ENABLED", True),
        "scheduler_alive": _automation_alive(),
        "cycle_running": _automation["running"],
        "interval_seconds": max(30, _live_int("AGENT_SCAN_INTERVAL_SECONDS", 900)),
        "early_morning_only": _live_bool("EARLY_MORNING_ONLY", True),
        "next_run_at": _automation["next_run_at"],
        "last_trigger": _automation["last_trigger"],
        "last_started_at": _automation["last_started_at"],
        "last_finished_at": _automation["last_finished_at"],
        "cycles_completed": _automation["cycles_completed"],
        "last_error": _automation["last_error"],
        "last_summary": _automation["last_summary"],
        "settings_revision": get_revision(),
    }


@app.post("/automation/trigger", tags=["Automation"])
async def automation_trigger():
    """Run one cycle right now, regardless of the timer."""
    summary = await _run_cycle_once("manual")
    return summary


class AutomationToggle(BaseModel):
    enabled: bool


@app.post("/automation/toggle", tags=["Automation"])
async def automation_toggle(request: AutomationToggle):
    """Persist the automation switch and start/stop the scheduler to match."""
    _get_settings_service().update({"SCAN_AUTOMATION_ENABLED": request.enabled})
    if request.enabled and not _automation_alive():
        _automation["task"] = asyncio.create_task(_automation_loop())
    elif not request.enabled and _automation_alive():
        _automation["task"].cancel()
        _automation["task"] = None
        _automation["next_run_at"] = None
    return await automation_status()


# ------------------------------------------------------------------
# Frontend static serving
# ------------------------------------------------------------------

from pathlib import Path

_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


def main():
    """Entry point for `ai-bettor` / `python -m backend.main`.

    Host, port and reload come from the environment so the same command serves
    local development (`RELOAD=true`) and production (bound to 127.0.0.1 behind
    Nginx). Auto-reload is off by default — under it uvicorn runs a supervisor
    plus a child process, which would give the scheduler two copies of itself.
    """
    import os

    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "").strip().lower() in ("1", "true", "yes", "on")
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()