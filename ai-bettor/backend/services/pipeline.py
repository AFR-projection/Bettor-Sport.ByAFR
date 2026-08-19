"""Agent pipeline orchestrator for AI Bettor.

Runs the full pipeline for each scanned match:
DATA → VALIDATION → NORMALIZATION → STATISTICAL MODEL → SIMULATION (repeated)
→ MARKET ANALYSIS → PROBABILITY → EV → RISK → MULTI-AGENT REVIEW
→ BETTOR BRAIN → BET / NO BET → SAVE → TELEGRAM (high-score picks only)

Pipeline flow:
1. Data Scout fetches matches from The Odds API (multi-key router, early-morning filter)
2. All raw data is saved to the database (matches, odds snapshots)
3. Quant + Simulation (multiple Monte Carlo batches) + Market + Risk
4. Bettor Brain decides BET / NO BET with 0-100 score
5. Only picks with score >= TELEGRAM_MIN_SCORE are sent to Telegram
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Dict, List, Optional

from backend.agents.bettor_brain import BettorBrain, get_better_brain
from backend.agents.data_scout import DataScout, DataScoutResult, get_data_scout
from backend.agents.market_analyst import MarketAnalyst, get_market_analyst
from backend.agents.quant_analyst import QuantAnalyst, get_quant_analyst
from backend.agents.risk_manager import RiskManager, get_risk_manager
from backend.agents.simulation_analyst import SimulationAnalyst, get_simulation_analyst
from backend.config import get_settings
from backend.database.models import (
    AgentAnalysis, Match, OddsSnapshot, Prediction, RiskAssessment, Simulation, SystemLog,
)
from backend.database.session import init_db, session_scope
from backend.integrations.telegram import TelegramNotifier, get_telegram_notifier
from backend.services.paper_betting import PaperBettingService, get_paper_betting
from backend.services.scoring import PickScoringEngine, get_scoring_engine

logger = logging.getLogger("ai-bettor.pipeline")


def _json_default(o: Any) -> Any:
    """JSON-safe fallback for numpy scalars and other non-serializable values."""
    if hasattr(o, "item"):
        try:
            return o.item()
        except (TypeError, ValueError):
            pass
    return str(o)


class AgentRuntime:
    """Tracks agent status for UI (status always from backend)."""

    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {
            name: {
                "status": "IDLE",
                "task": "Waiting",
                "last_execution": None,
                "execution_time": 0.0,
                "confidence": None,
                "error": None,
                "output": None,
                "timestamp": None,
            }
            for name in [
                "data_scout", "quant_analyst", "market_analyst",
                "simulation_analyst", "risk_manager", "bettor_brain",
            ]
        }

    def start(self, agent: str, task: str) -> None:
        self.agents[agent]["status"] = "ACTIVE"
        self.agents[agent]["task"] = task
        self.agents[agent]["error"] = None

    def finish(self, agent: str, output: Any, execution_time: float, confidence: Optional[int] = None) -> None:
        import datetime as dt
        now = dt.datetime.utcnow().isoformat()
        self.agents[agent]["status"] = "COMPLETE"
        self.agents[agent]["task"] = "Done"
        self.agents[agent]["last_execution"] = now
        self.agents[agent]["timestamp"] = now
        self.agents[agent]["execution_time"] = round(execution_time, 3)
        self.agents[agent]["confidence"] = confidence
        self.agents[agent]["output"] = output

    def fail(self, agent: str, error: str) -> None:
        import datetime as dt
        self.agents[agent]["status"] = "ERROR"
        self.agents[agent]["error"] = str(error)
        self.agents[agent]["timestamp"] = dt.datetime.utcnow().isoformat()

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return self.agents


class AiBettorPipeline:
    """Orchestrates all agents in the correct order."""

    def __init__(
        self,
        scout: Optional[DataScout] = None,
        quant: Optional[QuantAnalyst] = None,
        market: Optional[MarketAnalyst] = None,
        simulation: Optional[SimulationAnalyst] = None,
        risk: Optional[RiskManager] = None,
        brain: Optional[BettorBrain] = None,
        notifier: Optional[TelegramNotifier] = None,
        paper: Optional[PaperBettingService] = None,
        scoring: Optional[PickScoringEngine] = None,
    ):
        settings = get_settings()
        self.settings = settings
        self.scout = scout or get_data_scout()
        self.quant = quant or get_quant_analyst()
        self.market = market or get_market_analyst()
        self.simulation = simulation or get_simulation_analyst(
            simulations=settings.MONTE_CARLO_SIMULATIONS,
            random_seed=settings.RANDOM_SEED,
        )
        self.risk = risk or get_risk_manager()
        self.brain = brain or get_better_brain()
        self.notifier = notifier or get_telegram_notifier()
        self.paper = paper or get_paper_betting()
        self.scoring = scoring or get_scoring_engine()
        self.runtime = AgentRuntime()
        self.last_results: List[Dict[str, Any]] = []
        self.simulation_batches = settings.SIMULATION_BATCHES
        self.telegram_min_score = settings.TELEGRAM_MIN_SCORE
        self.telegram_max_picks = settings.TELEGRAM_MAX_PICKS

    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------

    def _persist_match(self, match: DataScoutResult) -> None:
        """Save match + odds snapshots to the database (idempotent)."""
        try:
            raw = match.raw_match_data or {}
            teams = match.normalized_data or {}
            kickoff = None
            if match.commence_time:
                try:
                    raw_ts = match.commence_time
                    if raw_ts.endswith("Z"):
                        raw_ts = raw_ts[:-1] + "+00:00"
                    kickoff = datetime.datetime.fromisoformat(raw_ts)
                except ValueError:
                    kickoff = None
            with session_scope() as session:
                existing = session.query(Match).filter(Match.match_id == match.match_id).first()
                if existing:
                    existing.home_team = teams.get("home_team", existing.home_team)
                    existing.away_team = teams.get("away_team", existing.away_team)
                    if kickoff:
                        existing.kickoff = kickoff
                else:
                    session.add(Match(
                        match_id=match.match_id,
                        home_team=teams.get("home_team", "UNKNOWN"),
                        away_team=teams.get("away_team", "UNKNOWN"),
                        kickoff=kickoff or datetime.datetime.utcnow(),
                        league=raw.get("sport_title", raw.get("league", "UNKNOWN")),
                        sport=raw.get("sport_key", "soccer"),
                        status="upcoming",
                    ))
                for bookmaker in raw.get("odds", []) or []:
                    bname = bookmaker.get("name", "UNKNOWN")
                    for market in bookmaker.get("markets", []) or []:
                        mkey = market.get("key", "UNKNOWN")
                        for sel in market.get("selections", []) or []:
                            try:
                                odd = float(sel.get("odd") or 0)
                            except (TypeError, ValueError):
                                continue
                            if odd <= 1:
                                continue
                            session.add(OddsSnapshot(
                                match_id=match.match_id,
                                bookmaker=bname[:50],
                                market=mkey[:50],
                                selection=str(sel.get("name", "UNKNOWN"))[:100],
                                line=str(market.get("point", "")),
                                odds=odd,
                            ))
        except Exception as e:
            logger.warning("Failed to persist match %s: %s", match.match_id, e)

    def _persist_agent_analysis(self, match_id: str, agent: str, status: str,
                                output: Any, exec_time: float, error: Optional[str] = None) -> None:
        try:
            import json as _json
            serialized = _json.dumps(output, default=_json_default) if output is not None else None
            with session_scope() as session:
                session.add(AgentAnalysis(
                    match_id=match_id,
                    agent_type=agent,
                    status=status,
                    output=serialized,
                    execution_time=exec_time,
                    error_message=error,
                ))
        except Exception as e:
            logger.warning("Failed to persist agent analysis: %s", e)

    def _persist_prediction(self, result: Dict[str, Any]) -> None:
        """Save a Bettor Brain decision as a Prediction row."""
        try:
            with session_scope() as session:
                session.add(Prediction(
                    match_id=result.get("match_id", "unknown"),
                    decision=result.get("decision", "NO BET"),
                    market=result.get("market"),
                    selection=result.get("selection"),
                    odds=result.get("odds"),
                    bookmaker=result.get("bookmaker"),
                    model_probability=result.get("probability", result.get("model_probability", 0)),
                    implied_probability=result.get("implied_probability", 0),
                    edge=result.get("edge", 0),
                    ev=result.get("ev", 0),
                    confidence_score=result.get("confidence", 0),
                    risk_level=result.get("risk", "UNKNOWN"),
                    reasoning=result.get("reasoning"),
                ))
        except Exception as e:
            logger.warning("Failed to persist prediction: %s", e)

    # ------------------------------------------------------------------
    # Scan cycle
    # ------------------------------------------------------------------

    def run_cycle(self, early_morning_only: Optional[bool] = None) -> Dict[str, Any]:
        """One full automated cycle: scan -> analyze -> save -> notify.

        Returns a summary dict. Safe to call repeatedly (scheduler / CLI).
        """
        t0 = time.time()
        self.runtime.start("data_scout", "Scanning matches from The Odds API (multi-key)")

        try:
            scan_results = self.scout.scan_matches(early_morning_only=early_morning_only)
        except Exception as e:
            self.runtime.fail("data_scout", str(e))
            logger.error("Scan failed: %s", e)
            return {"status": "error", "error": str(e), "matches_scanned": 0, "picks": []}

        self.runtime.finish(
            "data_scout",
            {
                "total_matches": len(scan_results),
                "early_morning_only": self.settings.EARLY_MORNING_ONLY,
                "results": [r.to_dict() for r in scan_results],
            },
            time.time() - t0,
        )

        if not scan_results:
            logger.info("No matches returned by API (or API not configured)")
            self._log("pipeline", "scan", "NO_MATCHES", scan_duration=time.time() - t0)
            return {"status": "empty", "matches_scanned": 0, "picks": []}

        picks: List[Dict[str, Any]] = []
        for match in scan_results:
            self._persist_match(match)
            decision = self.analyze_match(match)
            if not decision:
                continue
            self._persist_prediction(decision)
            if decision.get("decision") == "BET":
                score_result = self.scoring.score({
                    "model_probability": decision.get("probability", 0),
                    "ev": decision.get("ev", 0),
                    "edge": decision.get("edge", 0),
                    "simulation_stability": decision.get("simulation_stability", 0),
                    "odds": decision.get("odds", 0),
                    "data_quality": decision.get("data_quality", 50),
                    "risk_level": decision.get("risk", "UNKNOWN"),
                })
                decision["score"] = score_result["score"]
                decision["score_label"] = score_result["label"]
                if self.scoring.is_bettable(score_result):
                    picks.append(decision)
                    self.paper.place_bet(decision)
                else:
                    decision["decision"] = "NO BET"
                    decision["reasons"] = [*decision.get("reasons", []), f"SCORE_BELOW_THRESHOLD:{score_result['score']}"]

        # Rank by score, keep the best only
        picks.sort(key=lambda p: p.get("score", 0), reverse=True)
        self.last_results = picks

        # Telegram: only high-potential picks (score >= threshold), max N
        sent = 0
        qualified = [p for p in picks if (p.get("score") or 0) >= self.telegram_min_score]
        if qualified:
            top = qualified[: self.telegram_max_picks]
            for pick in top:
                if self.notifier.send_pick(pick):
                    sent += 1
            logger.info("Telegram: %s/%s high-score pick(s) sent", sent, len(top))
        else:
            self.notifier.send_no_bet_summary(
                f"{len(scan_results)} matches scanned (dini hari window), "
                f"no qualifying picks with score >= {self.telegram_min_score} today."
            )

        summary = {
            "status": "completed",
            "matches_scanned": len(scan_results),
            "bet_candidates": len(picks),
            "telegram_sent": sent,
            "duration_seconds": round(time.time() - t0, 2),
            "picks": picks,
        }
        self._log("pipeline", "scan", f"{len(scan_results)}_MATCHES_{len(picks)}_PICKS",
                  scan_duration=time.time() - t0)
        return summary

    def run_scan(self) -> List[Dict[str, Any]]:
        """Backwards-compatible wrapper: returns picks list."""
        summary = self.run_cycle()
        return summary.get("picks", [])

    def analyze_match(self, match: DataScoutResult) -> Optional[Dict[str, Any]]:
        """Run all agents on a single match and persist everything."""
        if not match or not match.raw_match_data:
            return None

        match_id = match.match_id
        odds_data = match.raw_match_data.get("odds", [])
        teams = match.normalized_data or {}
        home_team = teams.get("home_team", "UNKNOWN")
        away_team = teams.get("away_team", "UNKNOWN")

        # ---- QUANT ANALYST ----
        self.runtime.start("quant_analyst", f"Calculating probability for {match_id}")
        t = time.time()
        try:
            home_lambda = 1.5
            away_lambda = 1.2
            sim_pre = self.simulation.simulate(home_lambda, away_lambda)
            model_prob = sim_pre.home_win_probability
            quant_result = self.quant.analyze(match_id, model_prob, 1.91)
            self.runtime.finish("quant_analyst", quant_result.to_dict(), time.time() - t,
                                confidence=quant_result.confidence_score)
            self._persist_agent_analysis(match_id, "quant_analyst", "COMPLETE",
                                         quant_result.to_dict(), time.time() - t)
        except Exception as e:
            self.runtime.fail("quant_analyst", str(e))
            self._persist_agent_analysis(match_id, "quant_analyst", "ERROR", None, time.time() - t, str(e))
            return None

        # ---- MARKET ANALYST ----
        self.runtime.start("market_analyst", f"Comparing bookmakers for {match_id}")
        t = time.time()
        try:
            market_result = self.market.analyze(match_id, odds_data)
            self.runtime.finish("market_analyst", market_result.to_dict(), time.time() - t,
                                confidence=market_result.confidence)
            self._persist_agent_analysis(match_id, "market_analyst", "COMPLETE",
                                         market_result.to_dict(), time.time() - t)
        except Exception as e:
            self.runtime.fail("market_analyst", str(e))
            market_result = None
            self._persist_agent_analysis(match_id, "market_analyst", "ERROR", None, time.time() - t, str(e))

        # ---- SIMULATION ANALYST (repeated Monte Carlo batches) ----
        self.runtime.start("simulation_analyst",
                           f"Running {self.simulation.simulations:,} x {self.simulation_batches} simulations for {match_id}")
        t = time.time()
        try:
            sim_result = self.simulation.simulate(
                home_lambda, away_lambda,
                batches=self.simulation_batches,
            )
            self.runtime.finish("simulation_analyst", sim_result.to_dict(), time.time() - t)
            self._persist_agent_analysis(match_id, "simulation_analyst", "COMPLETE",
                                         sim_result.to_dict(), time.time() - t)
            try:
                with session_scope() as session:
                    session.add(Simulation(
                        match_id=match_id,
                        home_win_probability=sim_result.home_win_probability,
                        draw_probability=sim_result.draw_probability,
                        away_win_probability=sim_result.away_win_probability,
                        handicap_probability=sim_result.handicap_home_cover_probability,
                        over_probability=sim_result.over_25_probability,
                        under_probability=sim_result.under_25_probability,
                        variance=sim_result.variance,
                        stability=sim_result.stability,
                        simulation_count=sim_result.simulation_count,
                        random_seed=self.settings.RANDOM_SEED,
                    ))
            except Exception as e:
                logger.warning("Failed to persist simulation: %s", e)
        except Exception as e:
            self.runtime.fail("simulation_analyst", str(e))
            self._persist_agent_analysis(match_id, "simulation_analyst", "ERROR", None, time.time() - t, str(e))
            return None

        # ---- RISK MANAGER ----
        self.runtime.start("risk_manager", f"Risk assessment for {match_id}")
        t = time.time()
        try:
            exposure = self.paper.get_bankroll().get("total_staked", 0)
            bankroll = self.paper.get_bankroll().get("current_balance", 1000)
            risk_result = self.risk.assess(
                quant_result, sim_result, odds_data,
                current_exposure=exposure,
                bankroll=bankroll,
            )
            self.runtime.finish("risk_manager", risk_result.to_dict(), time.time() - t)
            self._persist_agent_analysis(match_id, "risk_manager", "COMPLETE",
                                         risk_result.to_dict(), time.time() - t)
            try:
                with session_scope() as session:
                    session.add(RiskAssessment(
                        match_id=match_id,
                        bankroll_risk_percent=(exposure / bankroll * 100.0) if bankroll else 0.0,
                        exposure=exposure,
                        drawdown=0.0,
                        correlation_risk=risk_result.correlation_concern,
                        risk_level=risk_result.risk_level,
                        veto_decision=risk_result.veto_decision,
                    ))
            except Exception as e:
                logger.warning("Failed to persist risk: %s", e)
        except Exception as e:
            self.runtime.fail("risk_manager", str(e))
            self._persist_agent_analysis(match_id, "risk_manager", "ERROR", None, time.time() - t, str(e))
            risk_result = None

        # ---- BETTOR BRAIN ----
        self.runtime.start("bettor_brain", f"Final decision for {match_id}")
        t = time.time()
        try:
            decision = self.brain.decide(
                quant_result.to_dict(),
                market_result.to_dict() if market_result else {},
                sim_result.to_dict(),
                risk_result.to_dict() if risk_result else {},
                {"home_team": home_team, "away_team": away_team, "match_id": match_id,
                 "kickoff": match.kickoff_wib},
                {"current_exposure": 0.0, "recent_results": [], "consecutive_losses": 0, "consecutive_wins": 0},
            )
            result = decision.to_dict()
            result["match_id"] = match_id
            result["home_team"] = home_team
            result["away_team"] = away_team
            result["data_quality"] = match.data_quality
            result["simulation_stability"] = sim_result.stability
            result["simulation_count"] = sim_result.simulation_count
            result["kickoff"] = match.kickoff_wib
            self._persist_prediction(result)
            self.runtime.finish("bettor_brain", result, time.time() - t, confidence=result.get("confidence"))
            self._persist_agent_analysis(match_id, "bettor_brain", "COMPLETE", result, time.time() - t)
            return result
        except Exception as e:
            self.runtime.fail("bettor_brain", str(e))
            self._persist_agent_analysis(match_id, "bettor_brain", "ERROR", None, time.time() - t, str(e))
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, service: str, action: str, status: str, **extra: Any) -> None:
        try:
            with session_scope() as session:
                session.add(SystemLog(
                    service=service,
                    agent=None,
                    match_id=None,
                    action=action,
                    status=status,
                    latency=extra.get("scan_duration"),
                    error_details=None,
                ))
        except Exception as e:
            logger.warning("Failed to write system log: %s", e)

    def agent_status(self) -> Dict[str, Dict[str, Any]]:
        return self.runtime.snapshot()

    def recent_picks(self) -> List[Dict[str, Any]]:
        return self.last_results


def get_pipeline() -> AiBettorPipeline:
    return AiBettorPipeline()