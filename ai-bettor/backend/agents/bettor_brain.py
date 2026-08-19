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

from typing import Any, Dict, List, Optional

from backend.models.probability_engine import (
    EVEngine,
    DataQualityChecker,
    ImpliedProbabilityEngine,
)


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
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "market": self.market,
            "selection": self.selection,
            "odds": self.odds,
            "bookmaker": self.bookmaker,
            "probability": self.probability,
            "implied_probability": self.implied_probability,
            "edge": self.edge,
            "ev": self.ev,
            "confidence": self.confidence,
            "risk": self.risk,
            "minimum_acceptable_odds": self.minimum_acceptable_odds,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "stake": self.stake,
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
    ):
        self.min_edge = min_edge
        self.min_ev = min_ev
        self.min_confidence = min_confidence
        self.min_acceptable_odds = min_acceptable_odds
        self.bankroll = bankroll
        self.risk_tolerance = risk_tolerance
        self.quality_checker = DataQualityChecker()
        self.ev_engine = EVEngine()
    
    def decide(self,
                quant_result: Dict[str, Any],
                market_result: Dict[str, Any],
                simulation_result: Dict[str, Any],
                risk_result: Dict[str, Any],
                match_data: Dict[str, Any],
                bettor_state: Dict[str, Any]) -> BettorBrainResult:
        """
        Make final betting decision.
        
        Integrates:
        - Quantitative analysis (probability, EV, edge)
        - Market analysis (best odds, consensus)
        - Simulation results (probability distribution, stability)
        - Risk assessment (data quality, exposure, drawdown)
        - Better behavioral state
        
        Returns BettorBrainResult with decision and details.
        """
        result = BettorBrainResult()
        
        # Extract key values from inputs
        model_probability = quant_result.get("model_probability", 0)
        edge = quant_result.get("edge", 0)
        ev = quant_result.get("ev", 0)
        confidence = quant_result.get("confidence_score", 0)
        
        best_odds = market_result.get("best_odds", 0)
        best_bookmaker = market_result.get("best_bookmaker", "")
        
        home_win_prob = simulation_result.get("home_win_probability", 0)
        draw_prob = simulation_result.get("draw_probability", 0)
        away_win_prob = simulation_result.get("away_win_probability", 0)
        
        simulation_stability = simulation_result.get("stability", 1.0)
        variance = simulation_result.get("variance", 10.0)
        
        data_quality = risk_result.get("data_quality_score", 100)
        veto_decision = risk_result.get("veto_decision", False)
        risk_level = risk_result.get("risk_level", "UNKNOWN")
        
        current_exposure = bettor_state.get("current_exposure", 0.0)
        recent_results = bettor_state.get("recent_results", [])
        consecutive_losses = bettor_state.get("consecutive_losses", 0)
        consecutive_wins = bettor_state.get("consecutive_wins", 0)
        
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
        
        # Check: Odds already bergerak melewati value threshold
        if best_odds < self.min_acceptable_odds:
            result.warnings.append("ODDS_BELOW_THRESHOLD")
        
        # Check: Minimum acceptable odds
        result.minimum_acceptable_odds = self.min_acceptable_odds
        
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
        
        # Determine market and selection
        market, selection = self._determine_market_selection(
            home_win_prob, draw_prob, away_win_prob,
            best_odds, best_bookmaker
        )
        
        result.market = market
        result.selection = selection
        result.odds = round(best_odds, 4)
        result.bookmaker = best_bookmaker
        result.probability = round(model_probability, 4)
        result.edge = round(recalculated_edge, 6)
        result.ev = round(ev_result["ev_per_unit"], 6)
        result.confidence = confidence
        result.risk = risk_level
        
        # Set stake based on bankroll and edge (Kelly-like calculation)
        stake = self._calculate_stake(
            model_probability, best_odds, edge, ev,
            current_exposure, self.bankroll
        )
        result.stake = stake
        result.potential_profit = round(stake * (best_odds - 1), 2)
        
        # Final decision
        result.decision = "BET"
        
        # Add reasons for BET
        result.reasons.append(f"POSITIVE_EDGE: {recalculated_edge:.4f}")
        result.reasons.append(f"POSITIVE_EV: {ev_result['ev_per_unit']:.4f}")
        result.reasons.append(f"CONFIDENCE: {confidence}/100")
        
        # Add minimum acceptable odds info
        result.reasons.append(f"MIN_ODDS: {self.min_acceptable_odds}")
        
        return result
    
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
                         bankroll: float) -> float:
        """
        Calculate stake amount.
        
        Uses a modified Kelly criterion approach,
        but capped and with risk controls.
        """
        if bankroll <= 0:
            return 0.0
        
        # Basic stake: proportional to edge and confidence
        # Kelly fraction = edge / (odds - 1) for decimal odds
        # But we use a fraction of Kelly for safety
        
        if odds <= 1 or model_prob <= 0 or model_prob >= 1:
            return 0.0
        
        # Kelly fraction
        net_odds = odds - 1
        if net_odds <= 0:
            return 0.0
        
        kelly_fraction = edge / net_odds if net_odds > 0 else 0
        
        # Cap Kelly at 0.25 (25% of bankroll max)
        kelly_fraction = min(kelly_fraction, 0.25)
        
        # Reduce if high exposure
        exposure_ratio = current_exposure / bankroll if bankroll > 0 else 0
        exposure_reduction = max(0, 1 - exposure_ratio * 2)  # Reduce 2% per 10% exposure
        kelly_fraction *= exposure_reduction
        
        # Reduce if low edge
        edge_reduction = max(0.5, edge * 10)  # Minimum 50% of calculated Kelly
        kelly_fraction *= edge_reduction
        
        # Final stake
        stake = bankroll * kelly_fraction
        
        # Minimum stake and maximum limits
        stake = max(1.0, min(stake, bankroll * 0.1))  # Min $1, max 10% bankroll
        
        return round(stake, 2)
    
    def _add_final_result(self, result: BettorBrainResult, decision: str):
        """Add final decision result."""
        result.decision = decision
    
    def quick_decide(self,
                     quant_result: Dict[str, Any],
                     market_result: Dict[str, Any],
                     simulation_result: Dict[str, Any],
                     risk_result: Dict[str, Any],
                     match_data: Dict[str, Any],
                     bettor_state: Dict[str, Any]) -> Dict[str, Any]:
        """Quick decision without full object construction."""
        result = self.decide(
            quant_result=quant_result,
            market_result=market_result,
            simulation_result=simulation_result,
            risk_result=risk_result,
            match_data=match_data,
            bettor_state=bettor_state,
        )
        return result.to_dict()


def get_better_brain(
    min_edge: float = 0.01,
    min_ev: float = 0.01,
    min_confidence: int = 60,
    min_acceptable_odds: float = 1.75,
    bankroll: float = 1000.0,
    risk_tolerance: str = "MEDIUM",
) -> BettorBrain:
    """Factory function to create BettorBrain instance."""
    return BettorBrain(
        min_edge=min_edge,
        min_ev=min_ev,
        min_confidence=min_confidence,
        min_acceptable_odds=min_acceptable_odds,
        bankroll=bankroll,
        risk_tolerance=risk_tolerance,
    )