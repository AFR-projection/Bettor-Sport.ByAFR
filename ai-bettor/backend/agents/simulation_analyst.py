"""Simulation Analyst Agent for AI Bettor.

Responsibilities:
- Run Monte Carlo simulation
- Calculate probability distributions
- Calculate probability 1X2, Handicap, Over/Under
- Calculate simulation stability
- Calculate variance

Configurable number of simulations (default 20000).
Uses configurable random seed for reproducibility.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from backend.models.probability_engine import (
    MonteCarloSimulationEngine,
    ProbabilityEnsemble,
)


class SimulationAnalystResult:
    """Structured output from Simulation Analyst agent."""
    
    def __init__(self):
        # Probability distributions
        self.home_win_probability: float = 0.0
        self.draw_probability: float = 0.0
        self.away_win_probability: float = 0.0
        self.over_25_probability: float = 0.0
        self.under_25_probability: float = 0.0
        self.handicap_home_cover_probability: float = 0.0
        self.handicap_away_cover_probability: float = 0.0
        
        # Simulation metrics
        self.variance: float = 0.0
        self.stability: float = 0.0
        self.simulation_count: int = 20000
        
        # Detailed
        self.mean_goal_difference: float = 0.0
        self.std_dev_goal_difference: float = 0.0
        
        # Per-market probabilities
        self.handicap_probabilities: Dict[str, float] = {}
        self.ou_probabilities: Dict[str, float] = {}
        
        self.warnings: List[str] = []
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "home_win_probability": self.home_win_probability,
            "draw_probability": self.draw_probability,
            "away_win_probability": self.away_win_probability,
            "over_25_probability": self.over_25_probability,
            "under_25_probability": self.under_25_probability,
            "handicap_home_cover_probability": self.handicap_home_cover_probability,
            "handicap_away_cover_probability": self.handicap_away_cover_probability,
            "variance": self.variance,
            "stability": self.stability,
            "simulation_count": self.simulation_count,
            "mean_goal_difference": self.mean_goal_difference,
            "std_dev_goal_difference": self.std_dev_goal_difference,
            "handicap_probabilities": self.handicap_probabilities,
            "ou_probabilities": self.ou_probabilities,
            "warnings": self.warnings,
        }


class SimulationAnalyst:
    """
    Simulation Analyst Agent - responsible for Monte Carlo simulations.
    
    Runs probability simulations to determine outcome distributions.
    Default: 20,000 simulations with configurable random seed.
    """
    
    def __init__(
        self,
        simulations: int = 20000,
        random_seed: Optional[int] = None,
    ):
        self.simulations = simulations
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.engine = MonteCarloSimulationEngine(
            simulations=simulations,
            random_seed=random_seed,
        )
    
    def simulate(self,
                 home_lambda: float,
                 away_lambda: float,
                 simulation_count: Optional[int] = None,
                 random_seed: Optional[int] = None,
                 batches: int = 1) -> SimulationAnalystResult:
        """
        Run Monte Carlo simulation for match outcomes.

        Runs `batches` independent simulation runs (each with its own seed)
        and averages the results for higher stability. Returns probability
        distribution across all markets.
        """
        n = simulation_count or self.simulations
        batches = max(1, batches)
        base_seed = random_seed if random_seed is not None else self.random_seed

        if batches <= 1:
            if base_seed is not None:
                np.random.seed(base_seed)
            sim_result = self.engine.simulate_match(
                home_lambda=home_lambda,
                away_lambda=away_lambda,
                simulations=n,
            )
        else:
            # Direct Poisson sampling per batch: each batch gets its own
            # seed (base_seed + i) so results are reproducible AND varied.
            agg = {k: 0.0 for k in (
                "home_win_probability", "draw_probability", "away_win_probability",
                "over_25_probability", "under_25_probability",
                "handicap_home_cover_probability", "handicap_away_cover_probability",
                "variance", "mean_goal_difference",
            )}
            for i in range(batches):
                if base_seed is not None:
                    np.random.seed(base_seed + i)
                home_goals = np.random.poisson(home_lambda, n)
                away_goals = np.random.poisson(away_lambda, n)
                gd = home_goals - away_goals
                total = home_goals + away_goals
                agg["home_win_probability"] += (gd > 0).mean()
                agg["draw_probability"] += (gd == 0).mean()
                agg["away_win_probability"] += (gd < 0).mean()
                agg["over_25_probability"] += (total > 2.5).mean()
                agg["under_25_probability"] += (total <= 2.5).mean()
                agg["handicap_home_cover_probability"] += (gd >= 0.5).mean()
                agg["handicap_away_cover_probability"] += (gd < 0.5).mean()
                agg["variance"] += float(gd.var())
                agg["mean_goal_difference"] += float(gd.mean())
            for k in agg:
                agg[k] /= batches
            sim_result = {
                **agg,
                "simulation_count": n * batches,
                "std_dev_goal_difference": math.sqrt(agg["variance"]),
            }
            # Engine not used for batches>1, so replicate its stability calc
            variance = agg["variance"]
            sim_result["stability"] = float(min(1.0, 5.0 / variance)) if variance > 0 else 1.0

        result = SimulationAnalystResult()
        
        # Map simulation results to output structure
        result.home_win_probability = sim_result["home_win_probability"]
        result.draw_probability = sim_result["draw_probability"]
        result.away_win_probability = sim_result["away_win_probability"]
        result.over_25_probability = sim_result["over_25_probability"]
        result.under_25_probability = sim_result["under_25_probability"]
        result.handicap_home_cover_probability = sim_result["handicap_home_cover_probability"]
        result.handicap_away_cover_probability = sim_result["handicap_away_cover_probability"]
        result.variance = sim_result["variance"]
        result.stability = sim_result["stability"]
        result.simulation_count = sim_result["simulation_count"]
        result.mean_goal_difference = sim_result["mean_goal_difference"]
        result.std_dev_goal_difference = sim_result["std_dev_goal_difference"]
        
        # Warnings for unstable simulations
        if sim_result["stability"] < 0.5:
            result.warnings.append("LOW_STABILITY")
        
        if sim_result["variance"] > 5.0:
            result.warnings.append("HIGH_VARIANCE")
        
        if n < self.simulations * 0.5:
            result.warnings.append("FEW_SIMULATIONS")
        
        return result
    
    def simulate_handicap(self,
                        home_lambda: float,
                        away_lambda: float,
                        handicap_line: float,
                        simulation_count: Optional[int] = None) -> Dict[str, float]:
        """
        Simulate Asian Handicap for given line.
        
        Returns handicap cover probabilities.
        """
        n = simulation_count or self.simulations
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        home_goals = np.random.poisson(home_lambda, n)
        away_goals = np.random.poisson(away_lambda, n)
        
        if handicap_line < 0:
            covered = np.sum(home_goals - away_goals >= abs(handicap_line))
        else:
            covered = np.sum(home_goals - away_goals > -handicap_line)
        
        covered_pct = covered / n
        uncovered_pct = 1 - covered_pct
        
        return {
            "home_handicap_cover": round(covered_pct, 6),
            "home_handicap_fail": round(uncovered_pct, 6),
        }
    
    def simulate_ou(self,
                    home_lambda: float,
                    away_lambda: float,
                    line: float = 2.5,
                    simulation_count: Optional[int] = None) -> Dict[str, float]:
        """
        Simulate Over/Under market.
        
        Returns over/under probabilities for given line.
        """
        n = simulation_count or self.simulations
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        home_goals = np.random.poisson(home_lambda, n)
        away_goals = np.random.poisson(away_lambda, n)
        
        total_goals = home_goals + away_goals
        
        over_line = (total_goals > line).sum() / n
        under_line = (total_goals <= line).sum() / n
        
        return {
            "over_probability": round(over_line, 6),
            "under_probability": round(under_line, 6),
        }
    
    def quick_simulate(self,
                       home_lambda: float,
                       away_lambda: float) -> Dict[str, Any]:
        """Quick simulation without full object construction."""
        result = self.simulate(
            home_lambda=home_lambda,
            away_lambda=away_lambda,
        )
        return result.to_dict()


def get_simulation_analyst(
    simulations: int = 20000,
    random_seed: Optional[int] = None,
) -> SimulationAnalyst:
    """Factory function to create SimulationAnalyst instance."""
    return SimulationAnalyst(
        simulations=simulations,
        random_seed=random_seed,
    )