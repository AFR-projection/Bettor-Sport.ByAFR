"""Expected Value engine."""
from __future__ import annotations

class EVEngine:
    @staticmethod
    def implied_prob(odds):
        return round(1.0/odds, 4) if odds > 0 else 0.0
    @staticmethod
    def normalize(probs):
        t = sum(probs.values())
        if t <= 0: return probs
        return {k: round(v/t,4) for k,v in probs.items()}
    @staticmethod
    def calc_ev(model_prob, odds):
        if odds <= 0 or model_prob < 0 or model_prob > 1: return 0.0
        return round(model_prob*(odds-1.0) - (1.0-model_prob)*1.0, 4)
    @staticmethod
    def calc_edge(model_prob, market_prob):
        if market_prob <= 0: return 0.0
        return round((model_prob - market_prob)/market_prob, 4)
    @staticmethod
    def fair_odds(model_prob):
        return round(1.0/model_prob, 2) if model_prob > 0 else 999.0
