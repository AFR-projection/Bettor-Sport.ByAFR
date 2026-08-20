"""Bettor Brain Agent for AI Bettor.

Final decision-maker that integrates all agent outputs.
Receives:
- Data Scout results
- Quant Analysis
- Market Analysis
- Simulation results
- Risk Assessment
- Odds
- Bankroll state
- Current exposure
- Historical performance
- Data quality

Outputs:
- BET or NO BET

Does NOT force betting. Only outputs BET when all conditions are met.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.models.probability_engine import (
    EVEngine,
    DataQualityChecker,
    ImpliedProbabilityEngine,
)

# Fallbacks used by `get_better_brain` when the settings service is unreachable.
BRAIN_DEFAULTS: Dict[str, Any] = {
    "MIN_EDGE": 0.02,
    "MIN_EV": 0.02,
    "MIN_CONFIDENCE": 60,
    "MIN_ODDS": 1.75,
    "MAX_ODDS": 6.0,
    "INITIAL_BANKROLL": 1000.0,
    "KELLY_FRACTION": 0.25,
    "MAX_STAKE_PERCENT": 2.0,
}


class BettorBrainResult:
    """Structured output from Bettor Brain agent."""
    
    def __init__(self):
        self.decision: str = "NO BET"  # BET or NO BET
        
        # Decision details
        self.market: str = ""
        self.selection: str = ""
        self.odds: float = 0.0
        self.bookmaker: str = ""
        
        # Probability & EV
        self.probability: float = 0.0  # model probability
        self.implied_probability: float = 0.0
        self.edge: float = 0.0
        self.ev: float = 0.0
        
        # Confidence & Risk
        self.confidence: int = 0  # 0-100
        self.risk: str = "UNKNOWN"
        
        # Constraints
        self.minimum_acceptable_odds: float = 0.0
        
        # Reasoning
        self.reasons: List[str] = []
        self.warnings: List[str] = []
        
        # Bet specifics
        self.stake: float = 0.0
        self.potential_profit: float = 0.0
        self.stake_percent: float = 0.0
        self.point: Optional[float] = None
        self.label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "market": self.market,
            "selection": self.selection,
            "point": self.point,
            "label": self.label,
            "odds": self.odds,
            "bookmaker": self.bookmaker,
            "probability": self.probability,
            "model_probability": self.probability,
            "implied_probability": self.implied_probability,
            "edge": self.edge,
            "ev": self.ev,
            "confidence": self.confidence,
            "risk": self.risk,
            "minimum_acceptable_odds": self.minimum_acceptable_odds,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "stake": self.stake,
            "stake_percent": self.stake_percent,
            "potential_profit": self.potential_profit,
        }


class BettorBrain:
    """
    Bettor Brain Agent - final decision-maker.
    
    Integrates all agent outputs and makes the final BET/NO BET decision.
    Follows professional bettor behavioral principles.
    """
    
    def __init__(
        self,
        min_edge: float = 0.01,
        min_ev: float = 0.01,
        min_confidence: int = 60,
        min_acceptable_odds: float = 1.75,
        bankroll: float = 1000.0,
        risk_tolerance: str = "MEDIUM",
        max_odds: float = 1000.0,
        kelly_fraction: float = 0.25,
        max_stake_percent: float = 2.0,
        min_stake: float = 1.0,
    ):
        self.min_edge = min_edge
        self.min_ev = min_ev
        self.min_confidence = min_confidence
        self.min_acceptable_odds = min_acceptable_odds
        self.max_odds = max_odds
        self.bankroll = bankroll
        self.risk_tolerance = risk_tolerance
        self.kelly_fraction = kelly_fraction
        self.max_stake_percent = max_stake_percent
        self.min_stake = min_stake
        self.quality_checker = DataQualityChecker()
        self.ev_engine = EVEngine()

    def decide(self,
                quant_result: Dict[str, Any],
                market_result: Dict[str, Any],
                simulation_result: Dict[str, Any],
                risk_result: Dict[str, Any],
                match_data: Dict[str, Any],
                bettor_state: Dict[str, Any],
                candidate: Optional[Dict[str, Any]] = None) -> BettorBrainResult:
        """
        Make final betting decision.

        Integrates:
        - Quantitative analysis (probability, EV, edge)
        - Market analysis (best odds, consensus)
        - Simulation results (probability distribution, stability)
        - Risk assessment (data quality, exposure, drawdown)
        - Better behavioral state

        `candidate` is the concrete opportunity picked by the market engine
        (market, selection, line, best price, blended probability). When it is
        supplied the brain judges *that* bet; without it the legacy 1X2 path is
        used and the market's best price is assumed to belong to the
        highest-probability outcome.

        Returns BettorBrainResult with decision and details.
        """
        result = BettorBrainResult()
        candidate = candidate or {}

        # Extract key values from inputs
        model_probability = quant_result.get("model_probability", 0)
        edge = quant_result.get("edge", 0)
        ev = quant_result.get("ev", 0)
        confidence = quant_result.get("confidence_score", 0)

        best_odds = market_result.get("best_odds", 0)
        best_bookmaker = market_result.get("best_bookmaker", "")
        if candidate:
            best_odds = candidate.get("odds", best_odds)
            best_bookmaker = candidate.get("bookmaker", best_bookmaker)
            if candidate.get("blended_probability") is not None:
                model_probability = candidate["blended_probability"]

        home_win_prob = simulation_result.get("home_win_probability", 0)
        draw_prob = simulation_result.get("draw_probability", 0)
        away_win_prob = simulation_result.get("away_win_probability", 0)

        simulation_stability = simulation_result.get("stability", 1.0)
        variance = simulation_result.get("variance", 10.0)

        data_quality = risk_result.get("data_quality_score", 100)
        veto_decision = risk_result.get("veto_decision", False)
        risk_level = risk_result.get("risk_level", "UNKNOWN")
        recommended_max_stake = risk_result.get("recommended_max_stake")

        current_exposure = bettor_state.get("current_exposure", 0.0)
        recent_results = bettor_state.get("recent_results", [])
        consecutive_losses = bettor_state.get("consecutive_losses", 0)
        consecutive_wins = bettor_state.get("consecutive_wins", 0)
        bankroll = float(bettor_state.get("bankroll") or self.bankroll)
        
        # ========= STEP 1: Behavioral checks =========
        
        # Check: Jangan mengejar kekalahan (chasing losses)
        if consecutive_losses >= 3 and ev <= 0:
            result.warnings.append("CHASING_LOSSES")
            result.decision = "NO BET"
            result.reasons.append("REJECTED: Chasing losses after consecutive losses")
            self._add_final_result(result, "NO BET")
            return result
        
        # Check: Jangan betting tanpa positive expected value
        if ev < self.min_ev:
            result.warnings.append("NEGATIVE_OR_ZERO_EV")
        
        # Check: Edge terlalu kecil
        if edge < self.min_edge:
            result.warnings.append("TOO_SMALL_EDGE")
        
        # Check: Jangan betting tanpa confidence yang cukup
        if confidence < self.min_confidence:
            result.warnings.append("LOW_CONFIDENCE")
        
        # Check: Data buruk
        if data_quality < 50:
            result.warnings.append("POOR_DATA_QUALITY")
            result.decision = "NO BET"
            result.reasons.append("REJECTED: Poor data quality")
            self._add_final_result(result, "NO BET")
            return result
        
        # Check: Risk Manager veto
        if veto_decision:
            result.warnings.append("RISK_MANAGER_VETO")
            result.decision = "NO BET"
            result.reasons.append("REJECTED: Risk Manager veto")
            self._add_final_result(result, "NO BET")
            return result
        
        # Check: Consecutive wins shouldn't increase risk
        if consecutive_wins > 5 and ev > 0:
            pass  # Intentional: no special penalty but also no bonus
        
        # ========= STEP 2: Market condition checks =========

        # Check: Minimum acceptable odds
        result.minimum_acceptable_odds = self.min_acceptable_odds

        # Check: no usable price at all. Without a real price every downstream
        # number (implied probability, edge, EV) is meaningless, so bail out
        # early with clean zeros instead of a fabricated edge.
        if not best_odds or best_odds <= 1.0:
            result.warnings.append("NO_PRICE")
            result.reasons.append("REJECTED: No usable price available")
            result.odds = 0.0
            result.probability = round(max(0.0, model_probability), 4)
            result.implied_probability = 0.0
            result.edge = 0.0
            result.ev = 0.0
            result.confidence = confidence
            result.risk = risk_level
            self._add_final_result(result, "NO BET")
            return result

        # Check: price above the ceiling we are willing to touch (longshots)
        if self.max_odds and best_odds > self.max_odds:
            result.warnings.append("ODDS_ABOVE_CEILING")
            result.reasons.append(
                f"REJECTED: Odds {best_odds:.2f} > maximum {self.max_odds}"
            )
            result.odds = round(best_odds, 4)
            result.bookmaker = best_bookmaker
            result.implied_probability = ImpliedProbabilityEngine.decimal_to_implied(best_odds)
            result.probability = round(max(0.0, model_probability), 4)
            result.confidence = confidence
            result.risk = risk_level
            self._add_final_result(result, "NO BET")
            return result

        # Check: Odds already bergerak melewati value threshold
        if best_odds < self.min_acceptable_odds:
            result.warnings.append("ODDS_BELOW_THRESHOLD")
        
        # ========= STEP 3: Probability evaluation =========
        
        # Determine which probability to use based on available data
        # Priority: model probability from quantitative engine
        # Fallback to simulation probabilities if model prob unavailable
        
        if model_probability <= 0:
            # Try to derive from simulation
            if home_win_prob > 0:
                # Use home win probability if no model prob
                model_probability = home_win_prob
            else:
                result.warnings.append("NO_MODEL_PROBABILITY")
                result.decision = "NO BET"
                result.reasons.append("REJECTED: No model probability available")
                self._add_final_result(result, "NO BET")
                return result
        
        # ========= STEP 4: Final decision logic =========
        
        # Calculate implied probability from odds
        implied_prob = ImpliedProbabilityEngine.decimal_to_implied(best_odds) if best_odds > 1 else 0
        result.implied_probability = implied_prob
        
        # Recalculate edge with implied probability
        recalculated_edge = round(model_probability - implied_prob, 6)
        result.edge = recalculated_edge
        
        # Calculate EV
        ev_result = self.ev_engine.calculate_ev(model_probability, best_odds)
        result.ev = ev_result["ev_per_unit"]
        
        # ========= STEP 5: Decision thresholds =========
        
        # NO BET conditions (check in order)
        
        # 1. Veto from risk manager already handled above
        
        # 2. Insufficient edge
        if recalculated_edge < self.min_edge:
            result.decision = "NO BET"
            result.reasons.append(f"NO BET: Edge {recalculated_edge:.4f} < minimum {self.min_edge}")
            self._add_final_result(result, "NO BET")
            return result
        
        # 3. Insufficient EV
        if ev_result["ev_per_unit"] < self.min_ev:
            result.decision = "NO BET"
            result.reasons.append(f"NO BET: EV {ev_result['ev_per_unit']:.4f} < minimum {self.min_ev}")
            self._add_final_result(result, "NO BET")
            return result
        
        # 4. Low confidence
        if confidence < self.min_confidence:
            result.decision = "NO BET"
            result.reasons.append(f"NO BET: Confidence {confidence} < minimum {self.min_confidence}")
            self._add_final_result(result, "NO BET")
            return result
        
        # 5. Simulation instability
        if simulation_stability < 0.5:
            result.decision = "NO BET"
            result.reasons.append(f"NO BET: Simulation instability {simulation_stability:.2f}")
            self._add_final_result(result, "NO BET")
            return result
        
        # 6. High variance
        if variance > 5.0:
            result.warnings.append("HIGH_SIMULATION_VARIANCE")
        
        # 7. Odds below minimum acceptable
        if best_odds < self.min_acceptable_odds:
            result.decision = "NO BET"
            result.reasons.append(f"NO BET: Odds {best_odds:.2f} < minimum acceptable {self.min_acceptable_odds}")
            self._add_final_result(result, "NO BET")
            return result
        
        # 8. Behavioral: no betting when chasing losses
        if consecutive_losses >= 3:
            result.warnings.append("BEHAVIORAL: Chasing losses detected")
        
        # 9. Behavioral: don't bet if recent all losses
        if len(recent_results or []) >= 3 and all(r == "loss" for r in recent_results[-3:]):
            if edge < 0.05:  # Only reject if edge isn't very strong
                result.decision = "NO BET"
                result.reasons.append("REJECTED: Recent all losses + modest edge")
                self._add_final_result(result, "NO BET")
                return result
        
        # ========= STEP 6: BET decision =========
        
        # If we reach here, all checks passed - make BET decision

        # Determine market and selection. A candidate from the market engine is
        # authoritative: it already knows which line and side the price belongs
        # to. The 1X2 fallback only applies when no candidate was supplied.
        if candidate:
            market = candidate.get("market", "1X2")
            selection = candidate.get("selection", "")
            result.point = candidate.get("point")
            result.label = candidate.get("label") or self._compose_label(
                market, selection, result.point
            )
        else:
            market, selection = self._determine_market_selection(
                home_win_prob, draw_prob, away_win_prob,
                best_odds, best_bookmaker
            )
            result.label = self._compose_label(market, selection, None)

        result.market = market
        result.selection = selection
        result.odds = round(best_odds, 4)
        result.bookmaker = best_bookmaker
        result.probability = round(model_probability, 4)
        result.edge = round(recalculated_edge, 6)
        result.ev = round(ev_result["ev_per_unit"], 6)
        result.confidence = confidence
        result.risk = risk_level

        # Set stake with fractional Kelly, capped by config and risk manager
        stake = self._calculate_stake(
            model_prob=model_probability,
            odds=best_odds,
            edge=recalculated_edge,
            ev=ev_result["ev_per_unit"],
            current_exposure=current_exposure,
            bankroll=bankroll,
            recommended_max_stake=recommended_max_stake,
        )
        result.stake = stake
        result.stake_percent = round(stake / bankroll * 100, 4) if bankroll > 0 else 0.0
        result.potential_profit = round(stake * (best_odds - 1), 2)

        if stake <= 0:
            result.warnings.append("STAKE_TOO_SMALL")
            result.reasons.append("REJECTED: Calculated stake below minimum")
            self._add_final_result(result, "NO BET")
            return result

        # Final decision
        result.decision = "BET"

        # Add reasons for BET
        result.reasons.append(f"POSITIVE_EDGE: {recalculated_edge:.4f}")
        result.reasons.append(f"POSITIVE_EV: {ev_result['ev_per_unit']:.4f}")
        result.reasons.append(f"CONFIDENCE: {confidence}/100")

        # Add minimum acceptable odds info
        result.reasons.append(f"MIN_ODDS: {self.min_acceptable_odds}")

        return result

    @staticmethod
    def _compose_label(market: str, selection: str, point: Optional[float]) -> str:
        """Human-readable bet label, e.g. `OU 2.5 Over` or `1X2 Home`."""
        parts = [str(market or "").strip()]
        if point is not None:
            parts.append(f"{float(point):+g}" if str(market).upper() == "HDP" else f"{float(point):g}")
        if selection:
            parts.append(str(selection).strip())
        return " ".join(p for p in parts if p)
    
    def _determine_market_selection(self,
                                    home_win_prob: float,
                                    draw_prob: float,
                                    away_win_prob: float,
                                    best_odds: float,
                                    best_bookmaker: str) -> Tuple[str, str]:
        """Determine which market and selection to bet on."""
        
        # Determine highest probability outcome
        probs = {
            "1X2 - Home": home_win_prob,
            "1X2 - Draw": draw_prob,
            "1X2 - Away": away_win_prob,
        }
        
        # Find the highest probability
        best_outcome = max(probs, key=probs.get)
        best_prob = probs[best_outcome]
        
        # Map to market key
        if best_prob == home_win_prob:
            market = "1X2"
            selection = "Home"
        elif best_prob == draw_prob:
            market = "1X2"
            selection = "Draw"
        else:
            market = "1X2"
            selection = "Away"
        
        return market, selection
    
    def _calculate_stake(self,
                         model_prob: float,
                         odds: float,
                         edge: float,
                         ev: float,
                         current_exposure: float,
                         bankroll: float,
                         recommended_max_stake: Optional[float] = None) -> float:
        """Fractional Kelly stake, capped by configuration and the risk manager.

        Full Kelly for a decimal-odds bet is `edge / (odds - 1)` where
        `edge = p - 1/odds`. We stake `kelly_fraction` of that (default a
        quarter), then apply, in order:

        - a hard ceiling of `max_stake_percent` of bankroll,
        - a linear taper as open exposure grows,
        - the risk manager's `recommended_max_stake`,
        - the `min_stake` floor (below it the bet is not worth placing).
        """
        if bankroll <= 0:
            return 0.0
        if odds <= 1 or model_prob <= 0 or model_prob >= 1:
            return 0.0

        net_odds = odds - 1
        if net_odds <= 0 or edge <= 0:
            return 0.0

        # Fractional Kelly on the true edge.
        full_kelly = edge / net_odds
        fraction = max(0.0, full_kelly) * max(0.0, self.kelly_fraction)

        # Hard ceiling from settings (percent of bankroll per bet).
        fraction = min(fraction, max(0.0, self.max_stake_percent) / 100.0)

        # Taper as open exposure builds up: at 50%+ exposure we stop staking.
        exposure_ratio = max(0.0, current_exposure) / bankroll
        fraction *= max(0.0, 1.0 - exposure_ratio * 2.0)

        stake = bankroll * fraction

        # Risk manager gets the last word on size.
        if recommended_max_stake is not None:
            try:
                stake = min(stake, max(0.0, float(recommended_max_stake)))
            except (TypeError, ValueError):
                pass

        stake = round(stake, 2)
        if stake < self.min_stake:
            # Only round up to the floor when the floor itself is affordable.
            ceiling = bankroll * max(0.0, self.max_stake_percent) / 100.0
            if self.min_stake <= ceiling and stake > 0:
                return round(float(self.min_stake), 2)
            return 0.0
        return stake
    
    def _add_final_result(self, result: BettorBrainResult, decision: str):
        """Add final decision result."""
        result.decision = decision
    
    def quick_decide(self,
                     quant_result: Dict[str, Any],
                     market_result: Dict[str, Any],
                     simulation_result: Dict[str, Any],
                     risk_result: Dict[str, Any],
                     match_data: Dict[str, Any],
                     bettor_state: Dict[str, Any],
                     candidate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Quick decision without full object construction."""
        result = self.decide(
            quant_result=quant_result,
            market_result=market_result,
            simulation_result=simulation_result,
            risk_result=risk_result,
            match_data=match_data,
            bettor_state=bettor_state,
            candidate=candidate,
        )
        return result.to_dict()


def get_better_brain(
    min_edge: Optional[float] = None,
    min_ev: Optional[float] = None,
    min_confidence: Optional[int] = None,
    min_acceptable_odds: Optional[float] = None,
    bankroll: Optional[float] = None,
    risk_tolerance: str = "MEDIUM",
    max_odds: Optional[float] = None,
    kelly_fraction: Optional[float] = None,
    max_stake_percent: Optional[float] = None,
) -> BettorBrain:
    """Build a BettorBrain from the live settings, with explicit overrides.

    Any argument left as `None` is read from the runtime settings service, so a
    threshold changed in the dashboard applies to the next decision without a
    restart. Falls back to the built-in defaults when settings are unreachable.
    """
    from backend.services.settings_service import get_setting

    def _pick(value: Any, key: str, cast):
        if value is not None:
            try:
                return cast(value)
            except (TypeError, ValueError):
                pass
        try:
            return cast(get_setting(key, BRAIN_DEFAULTS[key]))
        except (TypeError, ValueError):
            return cast(BRAIN_DEFAULTS[key])

    return BettorBrain(
        min_edge=_pick(min_edge, "MIN_EDGE", float),
        min_ev=_pick(min_ev, "MIN_EV", float),
        min_confidence=_pick(min_confidence, "MIN_CONFIDENCE", int),
        min_acceptable_odds=_pick(min_acceptable_odds, "MIN_ODDS", float),
        bankroll=_pick(bankroll, "INITIAL_BANKROLL", float),
        risk_tolerance=risk_tolerance,
        max_odds=_pick(max_odds, "MAX_ODDS", float),
        kelly_fraction=_pick(kelly_fraction, "KELLY_FRACTION", float),
        max_stake_percent=_pick(max_stake_percent, "MAX_STAKE_PERCENT", float),
    )