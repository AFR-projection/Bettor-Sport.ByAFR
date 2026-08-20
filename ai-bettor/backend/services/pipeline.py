"""Agent pipeline orchestrator for AI Bettor.

One cycle, per match:

DATA SCOUT (The Odds API v4, multi-key, dini-hari filter)
  -> MARKET ANALYST      de-vig every book, build a cross-book consensus
  -> POISSON CALIBRATION consensus 1X2 + O/U line  ->  expected goals (lambdas)
  -> SIMULATION ANALYST  repeated Monte Carlo batches for every market/line
  -> CANDIDATE BUILDER   best available price vs blended fair probability
  -> QUANT ANALYST       edge / EV / confidence for the single best candidate
  -> RISK MANAGER        data quality, exposure, drawdown, correlation, veto
  -> BETTOR BRAIN        BET / NO BET + fractional-Kelly stake
  -> SCORING             0-100, action threshold from settings
  -> LLM REVIEWER        optional sanity review (never a source of numbers)
  -> PERSIST             one Prediction row per match, plus every agent output
  -> TELEGRAM            only picks at or above TELEGRAM_MIN_SCORE

Nothing here invents a price. When the market data is too thin to build a
candidate the pipeline still records everything it computed and returns NO BET.

Every threshold is read from the live settings service, so a value saved from
the dashboard applies to the next cycle without restarting the process.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.bettor_brain import BettorBrain, get_better_brain
from backend.agents.data_scout import DataScout, DataScoutResult, get_data_scout
from backend.agents.market_analyst import MarketAnalyst, get_market_analyst
from backend.agents.quant_analyst import QuantAnalyst, get_quant_analyst
from backend.agents.risk_manager import RiskManager, get_risk_manager
from backend.agents.simulation_analyst import SimulationAnalyst, get_simulation_analyst
from backend.config import get_settings
from backend.core.market_math import Candidate, build_candidates, fit_lambdas
from backend.database.models import (
    AgentAnalysis, Match, OddsSnapshot, Prediction, RiskAssessment, Simulation, SystemLog,
)
from backend.database.session import session_scope
from backend.integrations.telegram import TelegramNotifier, get_telegram_notifier
from backend.services.paper_betting import PaperBettingService, get_paper_betting
from backend.services.scoring import PickScoringEngine, get_scoring_engine
from backend.services.settings_service import get_revision, get_setting

logger = logging.getLogger("ai-bettor.pipeline")

# Every agent the dashboard shows a status card for.
AGENT_KEYS = [
    "data_scout", "market_analyst", "simulation_analyst",
    "quant_analyst", "risk_manager", "bettor_brain", "llm_reviewer",
]

# Used when no candidate could be built: honest zeros instead of a fake edge.
ZERO_QUANT: Dict[str, Any] = {
    "model_probability": 0.0,
    "market_probability": 0.0,
    "edge": 0.0,
    "ev": 0.0,
    "confidence_score": 0,
    "risk_level": "UNKNOWN",
    "recommendation": "NO BET",
    "implied_probability": 0.0,
    "net_profit": 0.0,
    "probability_difference": 0.0,
}


def _json_safe(value: Any) -> Any:
    """Recursively convert a value into something the JSON column accepts."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result  # filter NaN


class AgentRuntime:
    """Tracks agent status for the dashboard (status always comes from here)."""

    @staticmethod
    def _blank() -> Dict[str, Any]:
        return {
            "status": "IDLE",
            "task": "Waiting",
            "last_execution": None,
            "execution_time": 0.0,
            "confidence": None,
            "error": None,
            "output": None,
            "timestamp": None,
        }

    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {name: self._blank() for name in AGENT_KEYS}

    def _slot(self, agent: str) -> Dict[str, Any]:
        """Never raise on an unknown agent name — just start tracking it."""
        if agent not in self.agents:
            self.agents[agent] = self._blank()
        return self.agents[agent]

    def start(self, agent: str, task: str) -> None:
        slot = self._slot(agent)
        slot["status"] = "ACTIVE"
        slot["task"] = task
        slot["error"] = None

    def finish(self, agent: str, output: Any, execution_time: float,
               confidence: Optional[int] = None) -> None:
        now = datetime.datetime.utcnow().isoformat()
        slot = self._slot(agent)
        slot["status"] = "COMPLETE"
        slot["task"] = "Done"
        slot["last_execution"] = now
        slot["timestamp"] = now
        slot["execution_time"] = round(execution_time, 3)
        slot["confidence"] = confidence
        slot["output"] = _json_safe(output)

    def skip(self, agent: str, reason: str) -> None:
        now = datetime.datetime.utcnow().isoformat()
        slot = self._slot(agent)
        slot["status"] = "SKIPPED"
        slot["task"] = reason
        slot["timestamp"] = now
        slot["error"] = None

    def fail(self, agent: str, error: str) -> None:
        slot = self._slot(agent)
        slot["status"] = "ERROR"
        slot["error"] = str(error)
        slot["timestamp"] = datetime.datetime.utcnow().isoformat()

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return self.agents


class AiBettorPipeline:
    """One full analysis cycle over every match the scout returns."""

    def __init__(self, scout: Optional[DataScout] = None,
                 quant: Optional[QuantAnalyst] = None,
                 market: Optional[MarketAnalyst] = None,
                 simulation: Optional[SimulationAnalyst] = None,
                 risk: Optional[RiskManager] = None,
                 brain: Optional[BettorBrain] = None,
                 notifier: Optional[TelegramNotifier] = None,
                 paper: Optional[PaperBettingService] = None,
                 scoring: Optional[PickScoringEngine] = None,
                 llm: Any = None):
        self.settings = get_settings()
        self.settings_revision = get_revision()
        self._load_thresholds()

        self.scout = scout or get_data_scout()
        self.market = market or get_market_analyst(
            default_total_goals=self.default_total_goals)
        self.simulation = simulation or get_simulation_analyst(
            simulations=self.monte_carlo_simulations,
            random_seed=self.random_seed,
        )
        self.quant = quant or get_quant_analyst(
            min_edge=self.min_edge,
            min_ev=self.min_ev,
            min_confidence=self.min_confidence,
        )
        self.risk = risk or get_risk_manager(
            min_data_quality=self.min_data_quality,
            max_uncertainty=self.max_uncertainty,
            veto_edge_threshold=self.min_edge,
            veto_ev_threshold=self.min_ev,
            max_exposure_percent=self.max_exposure_percent,
            max_stake_percent=self.max_stake_percent,
        )
        # The brain reads the live settings itself (edge/EV/odds/Kelly caps).
        self.brain = brain or get_better_brain()
        self.notifier = notifier or get_telegram_notifier()
        self.paper = paper or get_paper_betting()
        self.scoring = scoring or get_scoring_engine()
        self._llm = llm

        self.runtime = AgentRuntime()
        self.last_results: List[Dict[str, Any]] = []
        self.last_cycle: Optional[Dict[str, Any]] = None
        self._cycle_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Live configuration
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: Any) -> Any:
        """One setting: database value first, then .env snapshot, then default."""
        value = get_setting(key, getattr(self.settings, key, default))
        return default if value is None else value

    def _int(self, key: str, default: int) -> int:
        try:
            return int(float(self._cfg(key, default)))
        except (TypeError, ValueError):
            return default

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self._cfg(key, default))
        except (TypeError, ValueError):
            return default

    def _bool(self, key: str, default: bool) -> bool:
        value = self._cfg(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def _load_thresholds(self) -> None:
        """Read every tunable the cycle needs. Mutable, so tests can poke them."""
        self.monte_carlo_simulations = self._int("MONTE_CARLO_SIMULATIONS", 20000)
        self.simulation_batches = self._int("SIMULATION_BATCHES", 3)
        self.random_seed = self._int("RANDOM_SEED", 42)
        self.model_blend_weight = self._float("MODEL_BLEND_WEIGHT", 0.35)
        self.default_total_goals = self._float("DEFAULT_TOTAL_GOALS", 2.7)

        self.min_odds = self._float("MIN_ODDS", 1.75)
        self.max_odds = self._float("MAX_ODDS", 6.0)
        self.min_edge = self._float("MIN_EDGE", 0.02)
        self.min_ev = self._float("MIN_EV", 0.02)
        self.min_confidence = self._int("MIN_CONFIDENCE", 60)
        self.min_bookmakers = self._int("MIN_BOOKMAKERS", 3)
        self.min_data_quality = self._int("MIN_DATA_QUALITY", 50)
        self.max_uncertainty = self._float("MAX_UNCERTAINTY", 0.5)

        self.initial_bankroll = self._float("INITIAL_BANKROLL", 1000.0)
        self.max_stake_percent = self._float("MAX_STAKE_PERCENT", 2.0)
        self.max_exposure_percent = self._float("MAX_EXPOSURE_PERCENT", 20.0)

        self.early_morning_only = self._bool("EARLY_MORNING_ONLY", True)
        self.telegram_min_score = self._int("TELEGRAM_MIN_SCORE", 85)
        self.telegram_max_picks = self._int("TELEGRAM_MAX_PICKS", 5)
        self.telegram_send_no_bet = self._bool("TELEGRAM_SEND_NO_BET", True)
        self.llm_review_enabled = self._bool("LLM_REVIEW_ENABLED", True)

    def refresh_settings(self) -> bool:
        """Re-read the tunables. True when the settings revision had changed."""
        revision = get_revision()
        if revision == self.settings_revision:
            return False
        self.settings = get_settings()
        self.settings_revision = revision
        self._load_thresholds()
        self.scoring = get_scoring_engine()
        self.brain = get_better_brain()
        self.market = get_market_analyst(default_total_goals=self.default_total_goals)
        self.simulation = get_simulation_analyst(
            simulations=self.monte_carlo_simulations, random_seed=self.random_seed)
        self.quant = get_quant_analyst(
            min_edge=self.min_edge, min_ev=self.min_ev, min_confidence=self.min_confidence)
        self.risk = get_risk_manager(
            min_data_quality=self.min_data_quality,
            max_uncertainty=self.max_uncertainty,
            veto_edge_threshold=self.min_edge,
            veto_ev_threshold=self.min_ev,
            max_exposure_percent=self.max_exposure_percent,
            max_stake_percent=self.max_stake_percent,
        )
        logger.info("Pipeline rebuilt for settings revision %s", revision)
        return True

    def _llm_client(self) -> Any:
        """Lazily build the LLM reviewer (never fatal when it is unavailable)."""
        if self._llm is not None:
            return self._llm
        try:
            from backend.integrations.openrouter import get_llm_client
            self._llm = get_llm_client()
        except Exception as e:  # pragma: no cover - optional dependency
            logger.warning("LLM reviewer unavailable: %s", e)
            self._llm = None
        return self._llm

    def agent_status(self) -> Dict[str, Dict[str, Any]]:
        return self.runtime.snapshot()

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan(self, early_morning_only: Optional[bool] = None,
              sports: Optional[str] = None, regions: Optional[str] = None,
              markets: Optional[str] = None) -> List[DataScoutResult]:
        """Call the scout, forwarding only the kwargs it actually accepts."""
        kwargs: Dict[str, Any] = {"early_morning_only": early_morning_only}
        if sports:
            kwargs["sports"] = sports
        if regions:
            kwargs["regions"] = regions
        if markets:
            kwargs["markets"] = markets
        try:
            return list(self.scout.scan_matches(**kwargs) or [])
        except TypeError:
            # Minimal scouts (and the test doubles) only take early_morning_only.
            return list(self.scout.scan_matches(early_morning_only=early_morning_only) or [])

    def _has_odds_keys(self) -> bool:
        """True unless the scout says it has no API key to scan with.

        A scout that does not advertise `has_keys` (the test doubles) is always
        allowed to run, so the guard only ever fires on the real one.
        """
        has_keys = getattr(self.scout, "has_keys", None)
        return True if has_keys is None else bool(has_keys)

    def run_cycle(self, early_morning_only: Optional[bool] = None,
                  sports: Optional[str] = None, regions: Optional[str] = None,
                  markets: Optional[str] = None) -> Dict[str, Any]:
        """Scan, analyse every match, place paper bets, notify. Never raises."""
        started = time.time()
        started_at = datetime.datetime.utcnow().isoformat()
        errors: List[str] = []

        # A cycle without an Odds API key silently scans nothing, which reads
        # exactly like "the agent is not running". Say so instead.
        if not self._has_odds_keys():
            reason = ("no Odds API key configured "
                      "(add one under Settings > API Keys)")
            logger.warning("Cycle skipped: %s", reason)
            errors.append(f"data_scout: {reason}")
            self.runtime.skip("data_scout", reason)
            self._log_system("data_scout", None, "scan", "SKIPPED", 0.0, reason)
            return self._cycle_summary("no_api_key", started, started_at, 0, [], [], 0, errors)

        # STEP 1: DATA SCOUT
        self.runtime.start("data_scout", "Scanning odds")
        scan_started = time.time()
        matches: List[DataScoutResult] = []
        try:
            matches = self._scan(early_morning_only, sports, regions, markets)
            self.runtime.finish(
                "data_scout",
                {"matches": len(matches),
                 "match_ids": [m.match_id for m in matches][:50]},
                time.time() - scan_started,
            )
        except Exception as e:
            logger.exception("Data scout failed")
            errors.append(f"data_scout: {e}")
            self.runtime.fail("data_scout", e)
            self._log_system("data_scout", None, "scan", "ERROR", time.time() - scan_started, str(e))
            return self._cycle_summary("failed", started, started_at, 0, [], [], 0, errors)

        self._log_system("data_scout", None, "scan", "OK", time.time() - scan_started)

        # STEP 2: per-match analysis
        results: List[Dict[str, Any]] = []
        for match in matches:
            try:
                self._persist_match(match)
            except Exception as e:
                logger.warning("Could not persist match %s: %s", match.match_id, e)
                errors.append(f"persist {match.match_id}: {e}")
            try:
                decision = self.analyze_match(match)
            except Exception as e:
                logger.exception("Analysis failed for %s", match.match_id)
                errors.append(f"analyze {match.match_id}: {e}")
                continue
            if decision:
                results.append(decision)

        self.last_results = results
        picks = [r for r in results if r.get("decision") == "BET"]

        # STEP 3: paper bets for everything the brain approved
        for pick in picks:
            try:
                placement = self.paper.place_bet(pick)
                pick["bet"] = placement
            except Exception as e:
                logger.warning("Paper bet failed for %s: %s", pick.get("match_id"), e)
                errors.append(f"paper_bet {pick.get('match_id')}: {e}")

        # STEP 4: Telegram — only picks at or above the score threshold
        qualifying = sorted(
            (p for p in picks if _as_float(p.get("score")) >= self.telegram_min_score),
            key=lambda p: _as_float(p.get("score")), reverse=True,
        )[: max(0, self.telegram_max_picks)]
        telegram_sent = self._notify(qualifying, results, len(matches), errors)

        summary = self._cycle_summary(
            "completed", started, started_at, len(matches), results, picks,
            telegram_sent, errors,
        )
        self.last_cycle = summary
        return summary

    def _notify(self, qualifying: List[Dict[str, Any]], results: List[Dict[str, Any]],
                scanned: int, errors: List[str]) -> int:
        """Send qualifying picks, or one honest no-bet summary. Never raises."""
        sent = 0
        for pick in qualifying:
            try:
                if self.notifier.send_pick(pick):
                    sent += 1
            except Exception as e:
                logger.warning("Telegram send_pick failed: %s", e)
                errors.append(f"telegram: {e}")

        if sent == 0 and self.telegram_send_no_bet:
            best = max((_as_float(r.get("score")) for r in results), default=0.0)
            message = (
                f"Scan selesai: {scanned} match dianalisa, "
                f"{sum(1 for r in results if r.get('decision') == 'BET')} BET, "
                f"no qualifying picks at score >= {self.telegram_min_score} "
                f"(best score {best:g})."
            )
            try:
                self.notifier.send_no_bet_summary(message)
            except Exception as e:
                logger.warning("Telegram send_no_bet_summary failed: %s", e)
                errors.append(f"telegram: {e}")
        return sent

    def _cycle_summary(self, status: str, started: float, started_at: str,
                       scanned: int, results: List[Dict[str, Any]],
                       picks: List[Dict[str, Any]], telegram_sent: int,
                       errors: List[str]) -> Dict[str, Any]:
        return {
            "status": status,
            "started_at": started_at,
            "finished_at": datetime.datetime.utcnow().isoformat(),
            "duration_seconds": round(time.time() - started, 3),
            "matches_scanned": scanned,
            "matches_analyzed": len(results),
            "bet_candidates": len(picks),
            "picks": picks,
            "no_bets": [r for r in results if r.get("decision") != "BET"],
            "telegram_sent": telegram_sent,
            "telegram_min_score": self.telegram_min_score,
            "errors": errors,
            "settings_revision": self.settings_revision,
        }

    # ------------------------------------------------------------------
    # Per-match analysis
    # ------------------------------------------------------------------

    def analyze_match(self, match: DataScoutResult) -> Optional[Dict[str, Any]]:
        """Run the full agent chain for one match.

        Persists exactly one Prediction row (with the final decision), one
        Simulation row, one RiskAssessment row and one AgentAnalysis row per
        agent. Returns the decision dict, or None when the match is unusable.
        """
        if match is None or not getattr(match, "match_id", ""):
            return None

        match_id = match.match_id
        raw = getattr(match, "raw_match_data", None) or {}
        odds_data = raw.get("odds") or []
        home_team, away_team = self._teams(match)
        if not odds_data:
            logger.info("No odds for %s — nothing to analyse", match_id)
            return None

        self._ensure_match_row(match)
        self.runtime.finish(
            "data_scout",
            {"match_id": match_id, "data_quality": match.data_quality,
             "kickoff_wib": match.kickoff_wib, "warnings": match.warnings},
            0.0, int(_as_float(match.data_quality)),
        )

        # ===== MARKET ANALYST: de-vig every book, build the consensus =====
        self.runtime.start("market_analyst", f"Consensus for {match_id}")
        t0 = time.time()
        try:
            market_result = self.market.analyze(match_id, odds_data)
            groups = dict(self.market.market_groups)
            market_payload = market_result.to_dict()
            elapsed = time.time() - t0
            self.runtime.finish("market_analyst", market_payload, elapsed,
                                market_result.confidence)
            self._persist_agent_analysis(match_id, "market_analyst", "COMPLETE",
                                         market_payload, elapsed)
        except Exception as e:
            logger.exception("Market analyst failed for %s", match_id)
            self.runtime.fail("market_analyst", e)
            self._persist_agent_analysis(match_id, "market_analyst", "ERROR",
                                         None, time.time() - t0, str(e))
            return None

        fair = market_result.fair_probabilities or {}
        total_goals = market_result.total_goals_estimate
        home_lambda, away_lambda = fit_lambdas(
            fair.get("Home"), fair.get("Away"), total_goals,
            default_total=self.default_total_goals,
        )

        # ===== SIMULATION ANALYST: Monte Carlo, every market the books offer ==
        ou_lines = sorted({g.point for g in groups.values()
                           if g.market == "OU" and g.point is not None})
        hdp_lines = sorted({g.point for g in groups.values()
                            if g.market == "HDP" and g.point is not None})

        self.runtime.start("simulation_analyst", f"{self.simulation_batches} batch(es)")
        t0 = time.time()
        model_probabilities: Dict[str, float] = {}
        try:
            sim_result = self.simulation.simulate(
                home_lambda, away_lambda, batches=self.simulation_batches)
            model_probabilities = self.simulation.simulate_markets(
                home_lambda, away_lambda,
                ou_lines=list(ou_lines) or None,
                hdp_lines=list(hdp_lines) or None,
                batches=self.simulation_batches,
            )
            sim_payload = sim_result.to_dict()
            sim_payload["home_lambda"] = home_lambda
            sim_payload["away_lambda"] = away_lambda
            sim_payload["market_probabilities"] = model_probabilities
            elapsed = time.time() - t0
            self.runtime.finish("simulation_analyst", sim_payload, elapsed,
                                int(round(_as_float(sim_result.stability) * 100)))
            self._persist_agent_analysis(match_id, "simulation_analyst", "COMPLETE",
                                         sim_payload, elapsed)
            self._persist_simulation(match_id, sim_result)
        except Exception as e:
            logger.exception("Simulation failed for %s", match_id)
            self.runtime.fail("simulation_analyst", e)
            self._persist_agent_analysis(match_id, "simulation_analyst", "ERROR",
                                         None, time.time() - t0, str(e))
            return None

        # ===== CANDIDATE BUILDER: best price vs blended fair probability =====
        try:
            all_candidates = build_candidates(
                groups, model_probabilities,
                model_weight=self.model_blend_weight,
                min_books=self.min_bookmakers,
            )
        except Exception as e:
            logger.warning("Candidate build failed for %s: %s", match_id, e)
            all_candidates = []

        priced = [
            c for c in all_candidates
            if self.min_odds <= c.odds <= self.max_odds and c.ev > 0
        ]
        candidate: Optional[Candidate] = priced[0] if priced else None

        # ===== QUANT ANALYST: edge / EV / confidence for that one bet =====
        self.runtime.start("quant_analyst", "Edge and EV")
        t0 = time.time()
        if candidate is None:
            quant_payload = dict(ZERO_QUANT)
            self.runtime.skip("quant_analyst", "No value candidate")
            self._persist_agent_analysis(
                match_id, "quant_analyst", "SKIPPED",
                {"reason": "NO_VALUE_CANDIDATE",
                 "candidates_built": len(all_candidates),
                 "min_bookmakers": self.min_bookmakers,
                 "min_odds": self.min_odds, "max_odds": self.max_odds},
                time.time() - t0,
            )
        else:
            try:
                quant_result = self.quant.analyze(
                    match_id, candidate.blended_probability, candidate.odds)
                quant_payload = quant_result.to_dict()
                elapsed = time.time() - t0
                self.runtime.finish("quant_analyst", quant_payload, elapsed,
                                    int(_as_float(quant_payload.get("confidence_score"))))
                self._persist_agent_analysis(match_id, "quant_analyst", "COMPLETE",
                                            quant_payload, elapsed)
            except Exception as e:
                logger.exception("Quant analyst failed for %s", match_id)
                quant_payload = dict(ZERO_QUANT)
                self.runtime.fail("quant_analyst", e)
                self._persist_agent_analysis(match_id, "quant_analyst", "ERROR",
                                             None, time.time() - t0, str(e))

        # ===== RISK MANAGER: real exposure, drawdown, correlation, veto =====
        state = self._bettor_state(match_id)
        self.runtime.start("risk_manager", "Risk assessment")
        t0 = time.time()
        try:
            risk_result = self.risk.assess(
                quant_payload, sim_result, odds_data,
                current_exposure=state["current_exposure"],
                current_drawdown=state["current_drawdown"],
                bankroll=state["bankroll"],
                recent_results=state["recent_results"],
                data_quality=int(_as_float(match.data_quality)),
                market_dispersion=(candidate.dispersion if candidate
                                   else market_result.market_disagreement),
                book_count=(candidate.book_count if candidate
                            else market_result.bookmaker_count),
                open_bets_same_match=state["open_bets_same_match"],
            )
            risk_payload = risk_result.to_dict()
            elapsed = time.time() - t0
            self.runtime.finish("risk_manager", risk_payload, elapsed,
                                int(_as_float(risk_payload.get("data_quality_score"))))
            self._persist_agent_analysis(match_id, "risk_manager", "COMPLETE",
                                         risk_payload, elapsed)
        except Exception as e:
            logger.exception("Risk manager failed for %s", match_id)
            risk_payload = {
                "risk_level": "HIGH", "veto_decision": True,
                "veto_reason": f"RISK_ERROR: {e}", "data_quality_score": 0,
                "recommended_max_stake": 0.0, "warnings": ["RISK_ENGINE_ERROR"],
                "exposure_percent": 0.0, "drawdown_percent": 0.0,
                "uncertainty_score": 1.0, "correlation_concern": False,
            }
            self.runtime.fail("risk_manager", e)
            self._persist_agent_analysis(match_id, "risk_manager", "ERROR",
                                         None, time.time() - t0, str(e))
        self._persist_risk(match_id, risk_payload, state)

        # ===== BETTOR BRAIN: BET / NO BET + fractional-Kelly stake =====
        brain_market = {
            "best_odds": candidate.odds if candidate else 0.0,
            "best_bookmaker": candidate.bookmaker if candidate else "",
            "market_disagreement": (candidate.dispersion if candidate
                                    else market_result.market_disagreement),
            "bookmaker_count": (candidate.book_count if candidate
                                else market_result.bookmaker_count),
            "average_overround": market_result.average_overround,
            "line_movement_detected": market_result.line_movement_detected,
        }
        match_data = {
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "league": getattr(match, "league", "") or "",
            "kickoff": match.commence_time,
            "kickoff_wib": match.kickoff_wib,
            "data_quality": match.data_quality,
        }
        bettor_state = {
            "bankroll": state["bankroll"],
            "current_exposure": state["current_exposure"],
            "recent_results": state["recent_results"],
            "consecutive_losses": state["consecutive_losses"],
            "consecutive_wins": state["consecutive_wins"],
        }

        self.runtime.start("bettor_brain", "Final decision")
        t0 = time.time()
        try:
            decision = self.brain.decide(
                quant_payload, brain_market, sim_result.to_dict(), risk_payload,
                match_data, bettor_state,
                candidate=candidate.to_dict() if candidate else None,
            )
            result = decision.to_dict()
            brain_elapsed = time.time() - t0
        except Exception as e:
            logger.exception("Bettor brain failed for %s", match_id)
            self.runtime.fail("bettor_brain", e)
            self._persist_agent_analysis(match_id, "bettor_brain", "ERROR",
                                         None, time.time() - t0, str(e))
            return None

        # Context the dashboard, Telegram and the persistence layer all use.
        result["match_id"] = match_id
        result["home_team"] = home_team
        result["away_team"] = away_team
        result["match"] = f"{home_team} vs {away_team}".strip()
        result["league"] = match_data["league"]
        result["kickoff"] = match.commence_time
        result["kickoff_wib"] = match.kickoff_wib
        result["data_quality"] = int(_as_float(match.data_quality))
        result["candidate"] = candidate.to_dict() if candidate else None
        result["candidates_considered"] = len(all_candidates)
        result["candidates_priced"] = len(priced)
        result["alternatives"] = [c.to_dict() for c in priced[1:4]]
        result["home_lambda"] = home_lambda
        result["away_lambda"] = away_lambda
        result["total_goals_estimate"] = total_goals
        result["bookmaker_count"] = market_result.bookmaker_count
        result["markets_covered"] = market_result.markets_covered
        result["simulation_stability"] = _as_float(sim_result.stability)
        result["simulation_count"] = sim_result.simulation_count
        result["veto"] = bool(risk_payload.get("veto_decision"))
        result["veto_reason"] = risk_payload.get("veto_reason")
        result["exposure"] = state["current_exposure"]
        result["bankroll"] = state["bankroll"]
        if candidate is None:
            result["warnings"] = list(result.get("warnings") or []) + ["NO_VALUE_CANDIDATE"]

        # ===== SCORING: 0-100 and the configurable action threshold =====
        books_total = max(1, int(_as_float(market_result.bookmaker_count, 1.0)))
        score_input = {
            "model_probability": _as_float(result.get("probability")),
            "ev": _as_float(result.get("ev")),
            "edge": _as_float(result.get("edge")),
            "simulation_stability": _as_float(sim_result.stability),
            "market_disagreement": _as_float(brain_market["market_disagreement"]),
            "odds": _as_float(result.get("odds")),
            "minimum_acceptable_odds": _as_float(
                result.get("minimum_acceptable_odds"), self.min_odds),
            "data_quality": _as_float(risk_payload.get("data_quality_score")),
            "risk_level": result.get("risk") or "UNKNOWN",
            "bookmaker_consensus": (
                min(1.0, candidate.book_count / books_total) if candidate else 0.0),
        }
        score_result = self.scoring.score(score_input)
        result["score"] = score_result["score"]
        result["score_label"] = score_result["label"]
        result["score_factors"] = score_result["factors"]
        result["bet_threshold"] = score_result["bet_threshold"]

        # The score is a *gate*, never a source of a bet: it can only downgrade.
        if result["decision"] == "BET" and not self.scoring.is_bettable(score_result):
            result["decision"] = "NO BET"
            result["reasons"] = list(result.get("reasons") or []) + [
                f"REJECTED: Score {score_result['score']} below threshold "
                f"{score_result['bet_threshold']}"
            ]
            result["warnings"] = list(result.get("warnings") or []) + ["SCORE_BELOW_THRESHOLD"]
            result["stake"] = 0.0
            result["stake_percent"] = 0.0
            result["potential_profit"] = 0.0

        # ===== LLM REVIEWER: optional sanity check, never a number source =====
        result["llm_review"] = self._review(match_id, result, score_input)

        result["reasoning"] = " | ".join(str(r) for r in (result.get("reasons") or []))
        self.runtime.finish("bettor_brain", result, brain_elapsed,
                            int(_as_float(result.get("confidence"))))
        self._persist_agent_analysis(match_id, "bettor_brain", "COMPLETE",
                                     result, brain_elapsed)

        # ===== PERSIST: exactly one Prediction row, with the final decision ==
        self._persist_prediction(result)
        self._log_system("pipeline", match_id, "analyze", result["decision"])
        return result

    def _review(self, match_id: str, result: Dict[str, Any],
                score_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optional LLM sanity review. It can only downgrade a BET, never create one."""
        if result.get("decision") != "BET":
            self.runtime.skip("llm_reviewer", "No BET to review")
            return None
        if not self.llm_review_enabled:
            self.runtime.skip("llm_reviewer", "Disabled in settings")
            return None
        client = self._llm_client()
        if client is None or not getattr(client, "is_configured", False):
            self.runtime.skip("llm_reviewer", "No OpenRouter key configured")
            return None

        candidate = result.get("candidate") or {}
        context = {
            "match": result.get("match"),
            "kickoff_wib": result.get("kickoff_wib"),
            "bet": result.get("label"),
            "odds": result.get("odds"),
            "bookmaker": result.get("bookmaker"),
            "model_probability": result.get("probability"),
            "consensus_probability": candidate.get("consensus_probability"),
            "implied_probability": result.get("implied_probability"),
            "edge": result.get("edge"),
            "ev": result.get("ev"),
            "confidence": result.get("confidence"),
            "risk": result.get("risk"),
            "score": result.get("score"),
            "data_quality": score_input.get("data_quality"),
            "books_quoting": candidate.get("book_count"),
            "stake": result.get("stake"),
            "reasons": result.get("reasons"),
        }

        self.runtime.start("llm_reviewer", "Sanity review")
        t0 = time.time()
        try:
            review = client.analyze_match(context) or {}
        except Exception as e:
            logger.warning("LLM review failed for %s: %s", match_id, e)
            self.runtime.fail("llm_reviewer", e)
            self._persist_agent_analysis(match_id, "llm_reviewer", "ERROR", None,
                                         time.time() - t0, str(e))
            return None

        elapsed = time.time() - t0
        self.runtime.finish("llm_reviewer", review, elapsed)
        self._persist_agent_analysis(match_id, "llm_reviewer", "COMPLETE", review, elapsed)

        if review.get("agrees_with_quant") is False:
            concerns = ", ".join(str(c) for c in (review.get("concerns") or [])) or "unspecified"
            result["decision"] = "NO BET"
            result["reasons"] = list(result.get("reasons") or []) + [
                f"REJECTED: LLM review disagrees ({concerns})"
            ]
            result["warnings"] = list(result.get("warnings") or []) + ["LLM_REVIEW_DISAGREES"]
            result["stake"] = 0.0
            result["stake_percent"] = 0.0
            result["potential_profit"] = 0.0
        return review

    # ------------------------------------------------------------------
    # Bankroll / exposure state
    # ------------------------------------------------------------------

    @staticmethod
    def _teams(match: DataScoutResult) -> Tuple[str, str]:
        normalized = getattr(match, "normalized_data", None) or {}
        raw = getattr(match, "raw_match_data", None) or {}
        teams = raw.get("teams") or []
        home = normalized.get("home_team") or (teams[0] if len(teams) > 0 else "")
        away = normalized.get("away_team") or (teams[1] if len(teams) > 1 else "")
        return str(home or "UNKNOWN"), str(away or "UNKNOWN")

    def _bettor_state(self, match_id: str) -> Dict[str, Any]:
        """Real bankroll, real open exposure, real drawdown, real streaks."""
        bankroll_info: Dict[str, Any] = {}
        pending: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []
        try:
            bankroll_info = self.paper.get_bankroll() or {}
        except Exception as e:
            logger.warning("Bankroll unavailable: %s", e)
        try:
            pending = self.paper.pending_bets() or []
        except Exception as e:
            logger.warning("Pending bets unavailable: %s", e)
        try:
            history = self.paper.history(limit=25) or []
        except Exception as e:
            logger.warning("Bet history unavailable: %s", e)

        balance = _as_float(bankroll_info.get("current_balance"), self.initial_bankroll)
        # Money already committed to open bets: the exposure that actually matters.
        exposure = sum(_as_float(b.get("stake")) for b in pending)
        equity = balance + exposure
        drawdown = max(0.0, self.initial_bankroll - equity)

        outcomes = [str(b.get("result") or "").lower() for b in history if b.get("result")]
        consecutive_losses = 0
        for outcome in outcomes:  # history is newest first
            if outcome == "loss":
                consecutive_losses += 1
            else:
                break
        consecutive_wins = 0
        for outcome in outcomes:
            if outcome == "win":
                consecutive_wins += 1
            else:
                break

        return {
            "bankroll": balance if balance > 0 else self.initial_bankroll,
            "equity": round(equity, 2),
            "current_exposure": round(exposure, 2),
            "current_drawdown": round(drawdown, 2),
            "recent_results": outcomes[:10],
            "consecutive_losses": consecutive_losses,
            "consecutive_wins": consecutive_wins,
            "open_bets_same_match": sum(1 for b in pending if b.get("match_id") == match_id),
            "open_bets": len(pending),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dt(value: Any) -> datetime.datetime:
        """ISO string (or datetime) -> naive UTC datetime the DB accepts."""
        if isinstance(value, datetime.datetime):
            parsed = value
        else:
            text = str(value or "").strip().replace("Z", "+00:00")
            try:
                parsed = datetime.datetime.fromisoformat(text)
            except ValueError:
                parsed = datetime.datetime.utcnow()
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return parsed

    def _match_row_values(self, match: DataScoutResult) -> Dict[str, Any]:
        raw = getattr(match, "raw_match_data", None) or {}
        home_team, away_team = self._teams(match)
        return {
            "match_id": match.match_id,
            "home_team": home_team,
            "away_team": away_team,
            "kickoff": self._parse_dt(match.commence_time or raw.get("commence_time")),
            "league": str(getattr(match, "league", "") or raw.get("sport_title") or "UNKNOWN"),
            "sport": str(getattr(match, "sport_key", "") or raw.get("sport_key") or "soccer"),
            "status": "upcoming",
        }

    def _ensure_match_row(self, match: DataScoutResult) -> None:
        """Upsert the Match row only (no odds snapshots)."""
        values = self._match_row_values(match)
        try:
            with session_scope() as session:
                row = session.query(Match).filter(Match.match_id == values["match_id"]).first()
                if row is None:
                    session.add(Match(**values))
                else:
                    for key, value in values.items():
                        setattr(row, key, value)
        except Exception as e:
            logger.warning("Could not upsert match %s: %s", values["match_id"], e)

    def _persist_match(self, match: DataScoutResult) -> None:
        """Upsert the match and store one odds snapshot per selection."""
        self._ensure_match_row(match)
        raw = getattr(match, "raw_match_data", None) or {}
        rows: List[OddsSnapshot] = []
        for book in raw.get("odds") or []:
            if not isinstance(book, dict):
                continue
            bookmaker = str(book.get("name") or book.get("key") or "UNKNOWN")
            for market in book.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                market_key = str(market.get("key") or "UNKNOWN")
                for selection in market.get("selections") or []:
                    if not isinstance(selection, dict):
                        continue
                    odd = _as_float(selection.get("odd"))
                    if odd <= 1.0:
                        continue
                    point = selection.get("point")
                    if point is None:
                        point = market.get("point")
                    rows.append(OddsSnapshot(
                        match_id=match.match_id,
                        bookmaker=bookmaker[:50],
                        market=market_key[:50],
                        selection=str(selection.get("name") or "UNKNOWN")[:100],
                        line=None if point is None else f"{_as_float(point):g}",
                        odds=odd,
                    ))
        if not rows:
            return
        try:
            with session_scope() as session:
                session.add_all(rows)
        except Exception as e:
            logger.warning("Could not persist odds for %s: %s", match.match_id, e)

    def _persist_simulation(self, match_id: str, sim_result: Any) -> None:
        try:
            with session_scope() as session:
                session.add(Simulation(
                    match_id=match_id,
                    home_win_probability=_as_float(sim_result.home_win_probability),
                    draw_probability=_as_float(sim_result.draw_probability),
                    away_win_probability=_as_float(sim_result.away_win_probability),
                    handicap_probability=_as_float(sim_result.handicap_home_cover_probability),
                    over_probability=_as_float(sim_result.over_25_probability),
                    under_probability=_as_float(sim_result.under_25_probability),
                    variance=_as_float(sim_result.variance),
                    stability=_as_float(sim_result.stability),
                    simulation_count=int(_as_float(sim_result.simulation_count, 1.0)) or 1,
                    random_seed=self.random_seed,
                ))
        except Exception as e:
            logger.warning("Could not persist simulation for %s: %s", match_id, e)

    def _persist_risk(self, match_id: str, risk_payload: Dict[str, Any],
                      state: Dict[str, Any]) -> None:
        bankroll = state["bankroll"] or self.initial_bankroll
        max_stake = _as_float(risk_payload.get("recommended_max_stake"))
        open_same = int(state.get("open_bets_same_match") or 0)
        correlation = min(1.0, open_same * 0.5)
        if risk_payload.get("correlation_concern") and correlation == 0.0:
            correlation = 0.3
        try:
            with session_scope() as session:
                session.add(RiskAssessment(
                    match_id=match_id,
                    bankroll_risk_percent=round(max_stake / bankroll * 100, 4) if bankroll else 0.0,
                    exposure=_as_float(state.get("current_exposure")),
                    drawdown=_as_float(state.get("current_drawdown")),
                    correlation_risk=correlation,
                    risk_level=str(risk_payload.get("risk_level") or "UNKNOWN")[:20],
                    veto_decision=bool(risk_payload.get("veto_decision")),
                ))
        except Exception as e:
            logger.warning("Could not persist risk for %s: %s", match_id, e)

    def _persist_prediction(self, result: Dict[str, Any]) -> None:
        """One row per analysed match, holding the final decision."""
        try:
            with session_scope() as session:
                session.add(Prediction(
                    match_id=result.get("match_id"),
                    decision=str(result.get("decision") or "NO BET")[:10],
                    market=str(result.get("market") or "")[:50] or None,
                    selection=str(result.get("selection") or "")[:100] or None,
                    odds=_as_float(result.get("odds")),
                    bookmaker=str(result.get("bookmaker") or "")[:50] or None,
                    model_probability=_as_float(result.get("probability")),
                    implied_probability=_as_float(result.get("implied_probability")),
                    edge=_as_float(result.get("edge")),
                    ev=_as_float(result.get("ev")),
                    confidence_score=int(_as_float(result.get("confidence"))),
                    pick_score=_as_float(result.get("score")),
                    score_label=str(result.get("score_label") or "")[:30] or None,
                    risk_level=str(result.get("risk") or "UNKNOWN")[:20],
                    reasoning=result.get("reasoning") or None,
                ))
        except Exception as e:
            logger.warning("Could not persist prediction for %s: %s",
                           result.get("match_id"), e)

    def _persist_agent_analysis(self, match_id: str, agent: str, status: str,
                                output: Any, execution_time: float,
                                error: Optional[str] = None) -> None:
        """Store one agent's output. `AgentAnalysis.output` is a JSON column, so
        a dict goes in as a dict — never a pre-serialised string."""
        try:
            with session_scope() as session:
                session.add(AgentAnalysis(
                    match_id=match_id,
                    agent_type=agent[:50],
                    status=status[:50],
                    output=_json_safe(output) if output is not None else None,
                    execution_time=round(_as_float(execution_time), 4),
                    error_message=error,
                ))
        except Exception as e:
            logger.warning("Could not persist %s analysis for %s: %s", agent, match_id, e)

    def _log_system(self, service: str, match_id: Optional[str], action: str,
                    status: str, latency: Optional[float] = None,
                    error: Optional[str] = None) -> None:
        try:
            with session_scope() as session:
                session.add(SystemLog(
                    service=service[:50],
                    agent=service[:50],
                    match_id=match_id,
                    action=action[:100],
                    status=status[:50],
                    latency=None if latency is None else round(_as_float(latency), 4),
                    error_details=error,
                ))
        except Exception as e:
            logger.debug("Could not write system log: %s", e)


# ----------------------------------------------------------------------
# Process-wide pipeline, rebuilt whenever the settings revision changes
# ----------------------------------------------------------------------

_pipeline: Optional[AiBettorPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> AiBettorPipeline:
    """Shared pipeline. Rebuilt after a settings change so a value saved in the
    dashboard applies to the next cycle without restarting the process."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = AiBettorPipeline()
        else:
            _pipeline.refresh_settings()
        return _pipeline


def reset_pipeline() -> None:
    """Drop the cached pipeline (used by tests and after a settings reload)."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = None

