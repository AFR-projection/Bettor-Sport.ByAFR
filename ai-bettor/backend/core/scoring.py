"""Pick scoring engine 0-100."""
from __future__ import annotations; from dataclasses import dataclass

@dataclass
class ScoreWeights:
    model_prob: float = 0.20; ev: float = 0.20; edge: float = 0.15
    sim_stability: float = 0.15; market_agree: float = 0.10
    odds_quality: float = 0.05; data_quality: float = 0.10; uncertainty: float = 0.05
DEFAULT_WEIGHTS = ScoreWeights()

class ScoringEngine:
    def __init__(self, w=DEFAULT_WEIGHTS): self.w = w
    def score(self, d: dict):
        s = 0.0; mp = d.get("model_prob",0); ev = d.get("ev",0)
        ed = d.get("edge",0); st = d.get("sim_stability",0); ma = d.get("market_agree",0)
        oq = d.get("odds_quality",1); dq = d.get("data_quality",0); un = d.get("uncertainty",1)
        if mp > 0.65: s += self.w.model_prob * min(mp*1.5,1.0)
        elif mp > 0.5: s += self.w.model_prob * mp
        if ev > 0.1: s += self.w.ev * min(ev*5,1.0)
        elif ev > 0: s += self.w.ev * ev*2
        if ed > 0.05: s += self.w.edge * min(ed*20,1.0)
        elif ed > 0: s += self.w.edge * ed*5
        s += self.w.sim_stability * st
        s += self.w.market_agree * ma
        s += self.w.odds_quality * oq
        s += self.w.data_quality * dq
        s += self.w.uncertainty * (1-un)
        sc = round(min(s*100,100),1)
        if sc >= 90: tier = "PREMIUM"
        elif sc >= 80: tier = "BET_CANDIDATE"
        elif sc >= 70: tier = "WATCH"
        elif sc >= 60: tier = "PASS"
        else: tier = "NO_BET"
        return sc, tier
