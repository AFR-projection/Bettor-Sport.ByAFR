"""Agent pipeline orchestrator for AI Bettor.

Runs the full pipeline for each scanned match:
DATA → VALIDATION → NORMALIZATION → STATISTICAL MODEL → SIMULATION
→ MARKET ANALYSIS → PROBABILITY → EV → RISK → MULTI-AGENT REVIEW
→ BETTOR BRAIN → BET / NO BET → TELEGRAM
"""

from __future__ import annotations

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
from backend.integrations.telegram import TelegramNotifier, get_telegram_notifier
from backend.services.paper_betting import PaperBettingService, get_paper_betting
from backend.services.scoring import PickScoringEngine, get_scoring_engine

logger = logging.getLogger("ai-bettor.pipeline")


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
        import datetime
        now = datetime.datetime.utcnow().isoformat()
        self.agents[agent]["status"] = "COMPLETE"
        self.agents[agent]["task"] = "Done"
        self.agents[agent]["last_execution"] = now
        self.agents[agent]["timestamp"] = now
        self.agents[agent]["execution_time"] = round(execution_time, 3)
        self.agents[agent]["confidence"] = confidence
        self.agents[agent]["output"] = output

    def fail(self, agent: str, error: str) -> None:
        import datetime
        self.agents[agent]["status"] = "ERROR"
        self.agents[agent]["error"] = str(error)
        self.agents[agent]["timestamp"] = datetime.datetime.utcnow().isoformat()

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

    def run_scan(self) -> List[Dict[str, Any]]:
        """Full scan: fetch matches, run agents on each, notify Telegram."""
        t0 = time.time()
        self.runtime.start("data_scout", "Scanning matches from The Odds API")

        try:
            scan_results = self.scout.scan_matches()
        except Exception as e:
            self.runtime.fail("data_scout", str(e))
            logger.error("Scan failed: %s", e)
            return []

        self.runtime.finish(
            "data_scout",
            {"total_matches": len(scan_results), "results": [r.to_dict() for r in scan_results]},
            time.time() - t0,
        )

        if not scan_results:
            logger.info("No matches returned by API (or API not configured)")
            return []

        picks: List[Dict[str, Any]] = []
        for match in scan_results:
            decision = self.analyze_match(match)
            if decision and decision.get("decision") == "BET":
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

        # Rank picks and send only best to Telegram
        picks.sort(key=lambda p: p.get("score", 0), reverse=True)
        self.last_results = picks

        if picks:
            top = picks[:5]  # only best candidates to Telegram
            for pick in top:
                self.notifier.send_pick(pick)
            logger.info("Sent %s pick(s) to Telegram", len(top))
        else:
            self.notifier.send_no_bet_summary(
                f"{len(scan_results)} matches scanned, no qualifying picks today."
            )

        return picks

    def analyze_match(self, match: DataScoutResult) -> Optional[Dict[str, Any]]:
        """Run all agents on a single match."""
        if not match or not match.raw_match_data:
            return None

        match_id = match.match_id
        odds_data = match.raw_match_data.get("odds", [])

        # ---- QUANT ANALYST ----
        self.runtime.start("quant_analyst", f"Calculating probability for {match_id}")
        t = time.time()
        try:
            home_team = (match.normalized_data or {}).get("home_team", "UNKNOWN")
            away_team = (match.normalized_data or {}).get("away_team", "UNKNOWN")
            home_lambda = 1.5
            away_lambda = 1.2
            sim_pre = self.simulation.simulate(home_lambda, away_lambda)
            model_prob = sim_pre.home_win_probability
            quant_result = self.quant.analyze(match_id, model_prob, 1.91)
            self.runtime.finish("quant_analyst", quant_result.to_dict(), time.time() - t,
                                confidence=quant_result.confidence_score)
        except Exception as e:
            self.runtime.fail("quant_analyst", str(e))
            return None

        # ---- MARKET ANALYST ----
        self.runtime.start("market_analyst", f"Comparing bookmakers for {match_id}")
        t = time.time()
        try:
            market_result = self.market.analyze(match_id, odds_data)
            self.runtime.finish("market_analyst", market_result.to_dict(), time.time() - t,
                                confidence=market_result.confidence)
        except Exception as e:
            self.runtime.fail("market_analyst", str(e))
            market_result = None

        # ---- SIMULATION ANALYST ----
        self.runtime.start("simulation_analyst", f"Running {self.simulation.simulations:,} simulations for {match_id}")
        t = time.time()
        try:
            sim_result = self.simulation.simulate(home_lambda, away_lambda)
            self.runtime.finish("simulation_analyst", sim_result.to_dict(), time.time() - t)
        except Exception as e:
            self.runtime.fail("simulation_analyst", str(e))
            return None

        # ---- RISK MANAGER ----
        self.runtime.start("risk_manager", f"Risk assessment for {match_id}")
        t = time.time()
        try:
            risk_result = self.risk.assess(
                quant_result, sim_result, odds_data,
                current_exposure=self.paper.get_bankroll().get("total_staked", 0),
                bankroll=self.paper.get_bankroll().get("current_balance", 1000),
            )
            self.runtime.finish("risk_manager", risk_result.to_dict(), time.time() - t)
        except Exception as e:
            self.runtime.fail("risk_manager", str(e))
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
                {"home_team": home_team, "away_team": away_team, "match_id": match_id},
                {"current_exposure": 0.0, "recent_results": [], "consecutive_losses": 0, "consecutive_wins": 0},
            )
            result = decision.to_dict()
            result["match_id"] = match_id
            result["home_team"] = home_team
            result["away_team"] = away_team
            result["data_quality"] = match.data_quality
            result["simulation_stability"] = sim_result.stability
            result["simulation_count"] = sim_result.simulation_count
            self.runtime.finish("bettor_brain", result, time.time() - t, confidence=result.get("confidence"))
            return result
        except Exception as e:
            self.runtime.fail("bettor_brain", str(e))
            return None

    def agent_status(self) -> Dict[str, Dict[str, Any]]:
        return self.runtime.snapshot()

    def recent_picks(self) -> List[Dict[str, Any]]:
        return self.last_results


def get_pipeline() -> AiBettorPipeline:
    return AiBettorPipeline()