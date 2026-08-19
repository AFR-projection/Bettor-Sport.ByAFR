# """Statistical probability engine - Poisson and Elo models."""
from __future__ import annotations
import math
import numpy as np

class PoissonModel:
    @staticmethod
    def pmf(k: int, lam: float) -> float:
        if lam <= 0: return 1.0 if k == 0 else 0.0
        return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

    @staticmethod
    def match_probs(home_lam, away_lam, mx=10):
        hw, d, aw = 0.0, 0.0, 0.0
        for h in range(mx):
            for a in range(mx):
                p = PoissonModel.pmf(h, home_lam) * PoissonModel.pmf(a, away_lam)
                if h > a: hw += p
                elif h == a: d += p
                else: aw += p
        t = hw + d + aw
        if t > 0: hw /= t; d /= t; aw /= t
        return round(hw,4), round(d,4), round(aw,4)

class EloModel:
    def __init__(self, k=20, ha=65):
        self.k = k; self.ha = ha
    def expected(self, ra, rb):
        return 1.0 / (1.0 + 10**((rb - ra)/400.0))
    def win_probs(self, hr, ar):
        hw = self.expected(hr + self.ha, ar)
        aw = self.expected(ar, hr + self.ha)
        d = 1.0 - hw - aw
        if d < 0: hw -= abs(d)/2; aw -= abs(d)/2; d = 0
        return max(0,hw), max(0,d), max(0,aw)

class ProbabilityEngine:
    def __init__(self):
        self.poisson = PoissonModel()
        self.elo = EloModel()
    def calc_1x2(self, hl, al, use_elo=False, hr=0, ar=0):
        if use_elo and hr and ar: h,d,a = self.elo.win_probs(hr,ar)
        else: h,d,a = self.poisson.match_probs(hl, al)
        return {"home":round(h,4),"draw":round(d,4),"away":round(a,4)}
    def over_under(self, hl, al, line):
        mx = 15; over = 0.0
        for h in range(mx):
            for a in range(mx):
                p = PoissonModel.pmf(h,hl)*PoissonModel.pmf(a,al)
                if h+a > line: over += p
        return {"over":round(over,4),"under":round(1-over,4)}
