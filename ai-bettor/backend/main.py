"""Main FastAPI application for AI Bettor."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database.models import (
    AgentAnalysis, BankrollRecord, Bet, Match, OddsSnapshot,
    Prediction, RiskAssessment, Simulation, SystemLog,
)
from backend.database.session import get_db, init_db
from backend.services.backtest import Backtester, get_backtester
from backend.services.paper_betting import PaperBettingService, get_paper_betting
from backend.services.pipeline import AiBettorPipeline, get_pipeline
from backend.services.scoring import PickScoringEngine, get_scoring_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ai-bettor")

settings = get_settings()

app = FastAPI(
    title="AI Bettor",
    description="Autonomous AI sports betting analysis agent",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    database: str


class ScanRequest(BaseModel):
    sport: str = "soccer"
    regions: str = "idf"
    markets: str = "1X2,h2h,spreads,totals"
    odds_format: str = "decimal"


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
# Lazy singletons
# ------------------------------------------------------------------

_pipeline: Optional[AiBettorPipeline] = None
_paper: Optional[PaperBettingService] = None
_scoring: Optional[PickScoringEngine] = None


def _get_pipeline() -> AiBettorPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = get_pipeline()
    return _pipeline


def _get_paper() -> PaperBettingService:
    global _paper
    if _paper is None:
        _paper = get_paper_betting()
    return _paper


def _get_scoring() -> PickScoringEngine:
    global _scoring
    if _scoring is None:
        _scoring = get_scoring_engine()
    return _scoring


# ------------------------------------------------------------------
# System endpoints
# ------------------------------------------------------------------

@app.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check(db: Session = Depends(get_db)) -> HealthCheck:
    services: Dict[str, str] = {}
    services["the_odds_api"] = "configured" if settings.THE_ODDS_API_KEY else "missing_key"
    services["openrouter"] = "configured" if settings.OPENROUTER_API_KEY else "missing_key"
    services["telegram"] = "configured" if (settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID) else "missing_key"

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
        timestamp=__import__("datetime").datetime.utcnow().isoformat(),
        services=services,
        database=db_status,
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    return {
        "uptime_seconds": round(time.time() - _start_time),
        "agents": len(_get_pipeline().agent_status()),
        "mode": settings.BETTING_MODE,
        "timezone": settings.TIMEZONE,
        "monte_carlo_simulations": settings.MONTE_CARLO_SIMULATIONS,
        "random_seed": settings.RANDOM_SEED,
        "odds_poll_interval_seconds": settings.ODDS_POLL_INTERVAL_SECONDS,
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
    picks = pipeline.run_scan()
    return {
        "status": "completed",
        "matches_scanned": len(picks) if not picks else len(picks),
        "picks": picks,
        "agents": pipeline.agent_status(),
    }


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
    total_staked = sum(b.stake for b in bets)
    total_returned = sum(
        b.stake * (b.odds - 1) if b.result == "win" else (b.stake if b.result == "push" else 0)
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
        entry["staked"] += b.stake
        entry["profit"] += b.stake * (b.odds - 1) if b.result == "win" else (-b.stake if b.result == "loss" else 0)

    by_confidence: Dict[str, Dict[str, Any]] = {}
    for p in db.query(Prediction).all():
        key = "90+" if p.confidence_score >= 90 else "80-89" if p.confidence_score >= 80 else "70-79" if p.confidence_score >= 70 else "<70"
        entry = by_confidence.setdefault(key, {"predictions": 0})
        entry["predictions"] += 1

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / total_bets, 4) if total_bets else 0,
        "profit_loss": round(profit, 2),
        "roi": round(roi, 4),
        "total_staked": round(total_staked, 2),
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
# Startup
# ------------------------------------------------------------------

_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    logger.info("AI Bettor starting up...")
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database init failed: %s", e)
    logger.info("Timezone: %s", settings.TIMEZONE)
    logger.info("Monte Carlo simulations: %s", settings.MONTE_CARLO_SIMULATIONS)
    logger.info("Random seed: %s", settings.RANDOM_SEED)
    logger.info("Betting mode: %s", settings.BETTING_MODE)
    logger.info("AI Bettor ready.")


# ------------------------------------------------------------------
# Frontend static serving
# ------------------------------------------------------------------

from pathlib import Path

_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


def main():
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()