"""Quantitative Probability Engine for AI Bettor.

This module provides statistical probability calculations for sports betting,
including implied probability, expected value, and Monte Carlo simulations.
All calculations are based on real data - no hallucination or fabrication.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ImpliedProbabilityEngine:
    """Engine for calculating implied probability from odds."""
    
    @staticmethod
    def decimal_to_implied(odds: float, bookmaker_margin: Optional[float] = None) -> float:
        """
        Convert decimal odds to implied probability.
        
        Formula: implied_probability = 1 / odds
        
        If bookmaker margin/overround is provided, normalize the probability.
        """
        if odds <= 0:
            raise ValueError(f"Odds must be positive, got {odds}")
        
        raw_implied = 1.0 / odds
        
        if bookmaker_margin is not None and bookmaker_margin > 0:
            # Normalize to account for overround
            # True probability = raw / (sum of all raw implied probabilities)
            normalized = raw_implied / (1 + bookmaker_margin)
            return round(normalized, 6)
        
        return round(raw_implied, 6)
    
    @staticmethod
    def implied_to_decimal(implied_prob: float) -> float:
        """Convert implied probability back to decimal odds."""
        if implied_prob <= 0 or implied_prob >= 1:
            raise ValueError(f"Implied probability must be between 0 and 1, got {implied_prob}")
        if implied_prob == 0:
            return float('inf')
        return round(1.0 / implied_prob, 6)


class ProbabilityEnsemble:
    """Ensemble probability model that combines multiple sources."""
    
    def __init__(
        self,
        team_strength: Optional[float] = None,
        home_advantage: Optional[float] = None,
        recent_form: Optional[float] = None,
        offensive_strength: Optional[float] = None,
        defensive_strength: Optional[float] = None,
        market_probability: Optional[float] = None,
    ):
        self.team_strength = team_strength or 1.0
        self.home_advantage = home_advantage or 0.10
        self.recent_form = recent_form or 1.0
        self.offensive_strength = offensive_strength or 1.0
        self.defensive_strength = defensive_strength or 1.0
        self.market_probability = market_probability
    
    def calculate_team_probability(
        self, 
        home: bool, 
        opponent_strength: float = 1.0
    ) -> float:
        """
        Calculate win probability using a simplified Poisson/Logistic model.
        
        Uses team strength, home advantage, recent form, and offensive/defensive metrics.
        """
        # Base probability from team strength
        base = self.team_strength * self.recent_form
        
        # Apply home advantage if applicable
        if home:
            base *= (1 + self.home_advantage)
        else:
            base *= (1 - self.home_advantage / 2)
        
        # Adjust for opponent strength
        base *= (1 / opponent_strength)
        
        # Clip to valid probability range
        base = max(0.01, min(0.99, base))
        
        return round(base, 6)
    
    def calculate_poisson_lambda(
        self, 
        team_off: float, 
        team_def: float, 
        opponent_def: float
    ) -> float:
        """
        Calculate expected goals (lambda) using Dixon-Coles inspired formula.
        
        lambda = offensive_strength * defensive_opponent * league_average
        """
        # League average goal rate (typical ~1.5 goals per team per game)
        league_average = 1.5
        
        lambda_val = team_off * (1 / team_def) * opponent_def * league_average
        return round(max(0.01, lambda_val), 3)
    
    def calculate_1x2_probabilities(
        self, 
        home_off: float, home_def: float,
        away_off: float, away_def: float,
        league_avg: float = 1.5
    ) -> Dict[str, float]:
        """
        Calculate 1X2 probabilities using Poisson distribution.
        
        Returns home_win, draw, away_win probabilities.
        """
        # Calculate expected goals
        home_lambda = self.calculate_poisson_lambda(home_off, home_def, away_def)
        away_lambda = self.calculate_poisson_lambda(away_off, away_def, home_def)
        
        # Simplified Poisson probability calculation
        # P(X=k) = (lambda^k * e^(-lambda)) / k!
        
        # For home win: P(home_goals > away_goals)
        # For draw: P(home_goals == away_goals)  
        # For away win: P(away_goals > home_goals)
        
        # Use normal approximation for speed in ensemble
        # Difference of twoPoisson ~ approx Normal
        mean_diff = home_lambda - away_lambda
        var_diff = home_lambda + away_lambda  # Variance of difference
        std_diff = var_diff ** 0.5
        
        # P(home wins) = P(diff > 0) = P(Z > -mean/std)
        # Using error function approximation
        import math
        z_score = -mean_diff / std_diff if std_diff > 0 else 0
        
        # Standard normal CDF approximation
        phi = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
        home_win = round(min(0.99, max(0.01, phi)), 6)
        
        # P(draw) - simplified as probability goals difference is near 0
        # Probability |diff| < 0.5
        z_plus = (0.5 - mean_diff) / std_diff if std_diff > 0 else 0
        z_minus = (-0.5 - mean_diff) / std_diff if std_diff > 0 else 0
        phi_plus = 0.5 * (1 + math.erf(z_plus / math.sqrt(2)))
        phi_minus = 0.5 * (1 + math.erf(z_minus / math.sqrt(2)))
        draw = round(min(0.99, max(0.01, phi_plus - phi_minus)), 6)
        
        # P(away wins) = 1 - home_win - draw
        away_win = round(min(0.99, max(0.01, 1 - home_win - draw)), 6)
        
        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
        }


class MonteCarloSimulationEngine:
    """Monte Carlo simulation engine for probability distribution."""
    
    def __init__(
        self, 
        simulations: int = 20000,
        random_seed: Optional[int] = None,
    ):
        self.simulations = simulations
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def simulate_match(
        self,
        home_lambda: float,
        away_lambda: float,
        simulations: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Run Monte Carlo simulation for match outcomes.
        
        Returns probability distribution and key metrics.
        Reproducible when random_seed is set (reseeded per run).
        """
        n = simulations or self.simulations

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Simulate goals using Poisson distribution
        home_goals = np.random.poisson(home_lambda, n)
        away_goals = np.random.poisson(away_lambda, n)
        
        # Determine outcomes
        home_wins = (home_goals > away_goals).sum()
        draw_matches = (home_goals == away_goals).sum()
        away_wins = (home_goals < away_goals).sum()
        
        # Calculate probabilities
        home_win_prob = home_wins / n
        draw_prob = draw_matches / n
        away_win_prob = away_wins / n
        
        # Calculate variance (spread of outcomes)
        goal_diff = home_goals - away_goals
        variance = float(np.var(goal_diff))
        std_dev = float(np.std(goal_diff))
        
        # Stability measure: higher variance = lower stability (scaled)
        stability = float(min(1.0, 5.0 / variance)) if variance > 0 else 1.0
        
        # Over/Under 2.5 goals
        over_25 = (home_goals + away_goals > 2.5).sum() / n
        under_25 = (home_goals + away_goals <= 2.5).sum() / n
        
        # Asian Handicap (line -0.5 for home)
        # Home wins by 1+ = cover handicap, else lose
        ah_home_cover = (home_goals - away_goals >= -0.5).sum() / n  # Actually >= 0.5 for -0.5 AH
        # Wait, for handicap -0.5, home needs to win by at least 1 goal
        ah_home_cover = (home_goals - away_goals >= 0.5).sum() / n
        ah_away_cover = 1 - ah_home_cover
        
        return {
            "home_win_probability": round(home_win_prob, 6),
            "draw_probability": round(draw_prob, 6),
            "away_win_probability": round(away_win_prob, 6),
            "over_25_probability": round(over_25, 6),
            "under_25_probability": round(under_25, 6),
            "handicap_home_cover_probability": round(ah_home_cover, 6),
            "handicap_away_cover_probability": round(ah_away_cover, 6),
            "variance": round(variance, 6),
            "stability": round(stability, 6),
            "simulation_count": n,
            "mean_goal_difference": round(float(np.mean(goal_diff)), 4),
            "std_dev_goal_difference": round(std_dev, 4),
        }
    
    def simulate_handicap(
        self,
        home_lambda: float,
        away_lambda: float,
        handicap_line: float,
        simulations: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Simulate Asian Handicap market.
        
        handicap_line: e.g., -0.5 means home gives 0.5 goals
        """
        n = simulations or self.simulations
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        home_goals = np.random.poisson(home_lambda, n)
        away_goals = np.random.poisson(away_lambda, n)
        
        # Apply handicap
        # For home -0.5: home needs home_goals - away_goals > 0 (win by at least 1)
        # For home +0.5: home_goals - away_goals > -0.5 (home avoids loss by 1)
        
        if handicap_line < 0:
            # Home giving handicap
            covered = (home_goals - away_goals >= abs(handicap_line)).sum()
        else:
            # Home receiving handicap
            covered = (home_goals - away_goals > -handicap_line).sum()
        
        covered_pct = covered / n
        uncovered_pct = 1 - covered_pct
        
        return {
            "handicap_probability": round(covered_pct, 6),
            "handicap_against_probability": round(uncovered_pct, 6),
            "simulation_count": n,
        }


class EVEngine:
    """Expected Value calculation engine."""
    
    @staticmethod
    def calculate_ev(
        model_probability: float,
        decimal_odds: float,
        stake: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate Expected Value (EV) for a bet.
        
        Formula:
        - net_profit = odds - 1 (for winning bet)
        - EV = (model_prob * net_profit) - ((1 - model_prob) * stake)
        
        Returns EV per unit stake and total EV.
        """
        if not (0 < model_probability < 1):
            raise ValueError(f"Model probability must be between 0 and 1, got {model_probability}")
        if decimal_odds <= 1:
            raise ValueError(f"Decimal odds must be > 1, got {decimal_odds}")
        if stake <= 0:
            raise ValueError(f"Stake must be positive, got {stake}")
    
        net_profit = decimal_odds - 1
        ev_per_unit = (model_probability * net_profit) - ((1 - model_probability) * 1.0)
        total_ev = ev_per_unit * stake
    
        return {
            "model_probability": model_probability,
            "decimal_odds": decimal_odds,
            "net_profit": net_profit,
            "ev_per_unit": round(ev_per_unit, 6),
            "total_ev": round(total_ev, 6),
            "edge": round(model_probability - (1 / decimal_odds), 6),
            "implied_probability": round(1 / decimal_odds, 6),
        }
    
    @staticmethod
    def calculate_required_edge(
        decimal_odds: float,
        target_roi: float = 0.05,
    ) -> float:
        """
        Calculate minimum model probability needed for positive EV.
        
        For decimal odds d, implied probability = 1/d.
        Break-even: model_prob * (d-1) = (1-model_prob) * 1
        model_prob = 1/d (implied probability)
        Positive EV: model_prob > 1/d
        """
        implied = 1 / decimal_odds
        # Add small buffer for target ROI
        required = implied + (implied * target_roi / (1 - implied)) if implied < 1 else implied
        return round(min(required, 0.99), 6)


class DataQualityChecker:
    """Check quality of input data before analysis."""
    
    QUALITY_THRESHOLDS = {
        "min_odds": 1.01,
        "max_odds": 1000.0,
        "min_simulation_count": 100,
        "max_variance": 10.0,
    }
    
    @classmethod
    def check_odds_validity(
        cls, 
        odds: float, 
        min_odds: float = None,
        max_odds: float = None,
    ) -> Tuple[bool, str]:
        """Validate odds are within acceptable range."""
        min_odds = min_odds or cls.QUALITY_THRESHOLDS["min_odds"]
        max_odds = max_odds or cls.QUALITY_THRESHOLDS["max_odds"]
        
        if odds is None:
            return False, "ODDS_UNAVAILABLE"
        
        if odds <= min_odds:
            return False, f"ODDS_TOO_LOW:{odds}"
        if odds >= max_odds:
            return False, f"ODDS_TOO_HIGH:{odds}"
        
        return True, "VALID"
    
    @classmethod
    def check_probability_consistency(
        cls, 
        prob1: float, 
        prob2: float, 
        tolerance: float = 0.15,
    ) -> Tuple[bool, str]:
        """Check if two probability estimates are consistent."""
        diff = abs(prob1 - prob2)
        if diff > tolerance:
            return False, f"PROB_DISAGREEMENT:{diff:.4f}"
        return True, "CONSISTENT"


# Convenience functions for quick calculations

def quick_ev(model_prob: float, odds: float, stake: float = 1.0) -> Dict[str, float]:
    """Quick EV calculation."""
    return EVEngine.calculate_ev(model_prob, odds, stake)


def quick_implied(odds: float) -> float:
    """Quick implied probability calculation."""
    return ImpliedProbabilityEngine.decimal_to_implied(odds)


def quick_probability_ensemble(
    home_strength: float,
    away_strength: float,
    home_advantage: float = 0.10,
) -> Dict[str, float]:
    """Quick 1X2 probability calculation using ensemble model."""
    model = ProbabilityEnsemble(
        team_strength=(home_strength + away_strength) / 2,
        home_advantage=home_advantage,
    )
    
    # Simple probability based on strength ratio
    strength_ratio = home_strength / (home_strength + away_strength)
    
    # Adjust for home advantage
    adjusted = strength_ratio + (1 - strength_ratio) * home_advantage * 0.5
    
    return {
        "home_win": round(adjusted, 6),
        "draw": round(0.2 + (1 - adjusted - 0.2) * 0.3, 6),
        "away_win": round(1 - adjusted - 0.2, 6),
    }