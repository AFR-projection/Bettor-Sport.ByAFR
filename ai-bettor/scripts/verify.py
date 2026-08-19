import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
ERRORS = []
def check(name, fn):
    try:
        fn()
        print("[PASS] " + name)
    except Exception as e:
        ERRORS.append(name)
        print("[FAIL] " + name + ": " + str(e))
print("=== IMPORT TESTS ===")
from backend.config.settings import settings
print("Settings loaded OK")
from backend.database.db import init_db, close_engine, get_database_url
print("Database module OK")
from backend.core.probability_engine import ProbabilityEngine, PoissonModel, EloModel
print("ProbabilityEngine OK")
from backend.core.monte_carlo import MonteCarloEngine
print("MonteCarloEngine OK")
from backend.core.ev_engine import EVEngine
print("EVEngine OK")
from backend.core.scoring import ScoringEngine
print("ScoringEngine OK")
from backend.agents.data_scout import DataScoutAgent
from backend.agents.quant_analyst import QuantAnalystAgent
from backend.agents.market_analyst import MarketAnalystAgent
from backend.agents.simulation_analyst import SimulationAnalystAgent
from backend.agents.risk_manager import RiskManagerAgent
from backend.agents.bettor_brain import BettorBrainAgent, DecisionData, Decision
print("All 6 Agents OK")
from backend.integrations.odds_api import OddsAPIClient
from backend.integrations.openrouter import OpenRouterClient
from backend.integrations.telegram import TelegramClient
print("All Integrations OK")
from backend.services.scanner import ScannerService
from backend.services.pipeline import AnalysisPipeline
print("Services OK")
print()
print("=== ENGINE TESTS ===")
pe = ProbabilityEngine()
result = pe.calc_1x2(1.5, 1.2)
h = result["home"]; d = result["draw"]; a = result["away"]
print("1X2 prob: H=" + str(h) + " D=" + str(d) + " A=" + str(a) + " sum=" + str(round(h+d+a,4)))
assert abs(h + d + a - 1.0) < 0.01
ou = pe.over_under(1.5, 1.2, 2.5)
print("Over/Under: O=" + str(ou["over"]) + " U=" + str(ou["under"]))
mc = MonteCarloEngine(5000, 42)
r = mc.sim_1x2(1.5, 1.2)
print("MC 1X2: H=" + str(r["home_prob"]) + " D=" + str(r["draw_prob"]) + " A=" + str(r["away_prob"]) + " runs=" + str(r["total"]))
assert abs(r["home_prob"] + r["draw_prob"] + r["away_prob"] - 1.0) < 0.05
rt = mc.sim_totals(1.5, 1.2, 2.5)
print("Totals: O=" + str(rt["over"]) + " U=" + str(rt["under"]))
assert "over" in rt and "under" in rt
ee = EVEngine()
ip = ee.implied_prob(2.0)
assert ip == 0.5
ev = ee.calc_ev(0.55, 2.0)
print("EV for 55wd at 2.0: " + str(ev))
assert ev > 0
edge_val = ee.calc_edge(0.55, 0.5)
print("Edge: " + str(round(edge_val*100,1)) + "%")
assert edge_val > 0
fo = ee.fair_odds(0.55)
assert fo == 1.82
norm = ee.normalize({"a": 0.3, "b": 0.3, "c": 0.6})
assert abs(sum(norm.values()) - 1.0) < 0.001
print("EV Engine OK")
se = ScoringEngine()
score_hi, tier_hi = se.score({"model_prob": 0.9, "ev": 0.5, "edge": 0.15, "sim_stability": 0.9, "market_agree": 0.9, "odds_quality": 1, "data_quality": 0.9, "uncertainty": 0.9})
print("HDR pick: score=" + str(score_hi) + " tier=" + tier_hi)
assert tier_hi == "PREMIUM"
score_lo, tier_lo = se.score({"model_prob": 0.1, "ev": -0.5, "edge": -0.1, "sim_stability": 0.1, "market_agree": 0.1, "odds_quality": 0.1, "data_quality": 0.1, "uncertainty": 0.9})
print("LOW pick: score=" + str(score_lo) + " tier=" + tier_lo)
assert tier_lo == "NO_BET"
print("Scoring OK")
bb = BettorBrainAgent()
bad = DecisionData(ev=-0.1, edge=-0.01, confidence=20, data_quality=90, sim_stability=0.8, market_confidence=0.8, risk_level="LOW")
bad_dec = bb.decide(bad)
assert bad_dec.decision == "NO_BET"
print("Brain NO_BET logic OK")
good = DecisionData(ev=0.15, edge=0.08, confidence=75, data_quality=90, sim_stability=0.85, market_confidence=0.85, risk_level="LOW", match_id="1", market="1X2", selection="HOME", odds=2.0, bookmaker="TB", model_probability=0.65, implied_probability=0.5)
good_dec = bb.decide(good)
assert good_dec.decision == "BET"
print("Brain BET logic OK")
rm = RiskManagerAgent()
risk_lo = rm.assess("1", 80, 0.02, 0.05)
assert risk_lo.veto == False
print("Low risk: " + risk_lo.risk_level)
risk_hi = rm.assess("2", 20, 0.2, 0.5)
assert risk_hi.veto == True
print("High risk vetoed: " + risk_hi.veto_reason)
sim = SimulationAnalystAgent(2000)
md = {"api_match_id": "test", "home_lambda": 1.5, "away_lambda": 1.2}
s1x2 = sim.analyze_1x2(md)
assert s1x2.status == "COMPLETE"
print("Sim 1X2: H=" + str(s1x2.home_win_prob) + " D=" + str(s1x2.draw_prob) + " A=" + str(s1x2.away_win_prob))
sh = sim.analyze_handicap(md, -0.5)
st = sim.analyze_totals(md, 2.5)
print("Sim totals: O=" + str(st.over_prob) + " U=" + str(st.under_prob))
scout = DataScoutAgent()
match_data = {"api_match_id": "m1", "commence_time": "2026-12-01T10:00:00Z"}
odds_data = [{"bookmaker":"BM1","market":"h2h","selection":"Home","odds":2.0,"data_quality":100},{"bookmaker":"BM2","market":"h2h","selection":"Away","odds":3.0,"data_quality":90},{"bookmaker":"BM3","market":"h2h","selection":"Draw","odds":3.5,"data_quality":85}]
sr = scout.scout(match_data, odds_data)
print("Data quality: " + str(sr.data_quality) + ", Books: " + str(sr.bookmakers))
print()
print("=== ALL TESTS PASSED ===")
print("Errors: " + str(len(ERRORS)))
if ERRORS: print("FAILED: " + str(ERRORS))

