"""Monte Carlo simulation engine."""
from __future__ import annotations; import numpy as np

class MonteCarloEngine:
    def __init__(self, n_sim=20000, seed=42):
        self.n = n_sim; self.seed = seed
        self.rng = np.random.default_rng(seed if seed else None)
    def sim_1x2(self, hl, al):
        hg = self.rng.poisson(hl, self.n); ag = self.rng.poisson(al, self.n)
        hw = int(np.sum(hg > ag)); dr = int(np.sum(hg == ag)); aw = int(np.sum(hg < ag))
        t = float(self.n); d = np.var(hg - ag)
        return {"home_prob":round(hw/t,4),"draw_prob":round(dr/t,4),"away_prob":round(aw/t,4),"variance":round(float(d),4),"stability":round(min(hw/t,dr/t,aw/t)*4,4),"total":self.n}
    def sim_handicap(self, hl, al, hc):
        hg = self.rng.poisson(hl, self.n); ag = self.rng.poisson(al, self.n)
        diff = hg - ag + hc; wins = int(np.sum(diff > 0))
        return {"prob":round(wins/self.n,4),"total":self.n,"variance":round(float(np.var(diff)),4)}
    def sim_totals(self, hl, al, line):
        hg = self.rng.poisson(hl, self.n); ag = self.rng.poisson(al, self.n)
        tot = hg + ag; over = int(np.sum(tot > line))
        return {"over":round(over/self.n,4),"under":round(1-over/self.n,4),"total":self.n,"variance":round(float(np.var(tot)),4),"stability":round(min(over,self.n-over)/self.n*4,4)}
