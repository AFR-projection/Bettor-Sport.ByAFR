"""Tests for the quantitative market core in backend/core/market_math.py.

This module is where the edge actually comes from (per-book de-vig, cross-book
consensus, exact Poisson grids, line shopping), so the maths is pinned down
here rather than only exercised indirectly through the pipeline.
"""

from __future__ import annotations

import pytest

from backend.core.market_math import (
    blend_probability,
    build_candidates,
    collect_market_groups,
    devig_proportional,
    fit_lambdas,
    market_totals_estimate,
    overround,
    poisson_1x2,
    poisson_handicap_cover,
    poisson_over,
    poisson_pmf,
    solve_total_goals,
)


def book(name: str, market: str, selections: list, point=None) -> dict:
    """Build one bookmaker entry in the canonical odds structure."""
    return {
        "name": name,
        "markets": [{
            "key": market,
            "point": point,
            "selections": [
                {"name": sel, "odd": odd, "point": point} for sel, odd in selections
            ],
        }],
    }


# ------------------------------------------------------------------
# Poisson helpers
# ------------------------------------------------------------------

class TestPoisson:
    def test_pmf_is_a_distribution(self):
        total = sum(poisson_pmf(k, 1.4) for k in range(0, 40))
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_pmf_zero_lambda(self):
        assert poisson_pmf(0, 0.0) == 1.0
        assert poisson_pmf(1, 0.0) == 0.0

    def test_1x2_sums_to_one(self):
        home, draw, away = poisson_1x2(1.6, 1.1)
        assert home + draw + away == pytest.approx(1.0, abs=1e-6)
        assert home > away  # stronger attack wins more often

    def test_1x2_symmetric_lambdas(self):
        home, draw, away = poisson_1x2(1.3, 1.3)
        assert home == pytest.approx(away, abs=1e-9)
        assert 0.2 < draw < 0.35

    def test_over_decreases_with_line(self):
        p25 = poisson_over(1.35, 1.35, 2.5)
        p35 = poisson_over(1.35, 1.35, 3.5)
        assert 0.0 < p35 < p25 < 1.0

    def test_over_matches_total_lambda_only(self):
        # Total goals depend on the sum of the lambdas, not their split.
        assert poisson_over(2.0, 0.7, 2.5) == pytest.approx(
            poisson_over(1.35, 1.35, 2.5), abs=1e-9)

    def test_handicap_cover_excludes_push(self):
        # A level handicap between equal teams is a coin flip once the pushes
        # (draws) are removed from the sample space.
        assert poisson_handicap_cover(1.3, 1.3, 0.0) == pytest.approx(0.5, abs=1e-6)

    def test_handicap_giving_start_is_harder(self):
        assert poisson_handicap_cover(1.6, 1.1, -0.5) < poisson_handicap_cover(1.6, 1.1, 0.5)


class TestLambdaFitting:
    def test_solve_total_goals_round_trip(self):
        p_over = poisson_over(1.4, 1.4, 2.5)
        assert solve_total_goals(p_over, 2.5) == pytest.approx(2.8, abs=0.02)

    def test_solve_total_goals_rejects_extremes(self):
        assert solve_total_goals(0.0, 2.5) is None
        assert solve_total_goals(1.0, 2.5) is None

    def test_fit_lambdas_conserves_total(self):
        home, away = fit_lambdas(0.55, 0.22, 3.1)
        assert home + away == pytest.approx(3.1, abs=0.01)
        assert home > away

    def test_fit_lambdas_falls_back_symmetrically(self):
        # No usable market probability must not invent a favourite.
        home, away = fit_lambdas(None, None, None, default_total=2.7)
        assert home == away == pytest.approx(1.35, abs=1e-6)

    def test_fit_lambdas_clamps_absurd_totals(self):
        home, away = fit_lambdas(0.5, 0.3, 99.0)
        assert home + away <= 7.0


# ------------------------------------------------------------------
# De-vig / overround
# ------------------------------------------------------------------

class TestDevig:
    def test_devig_sums_to_one(self):
        fair = devig_proportional({"Home": 2.10, "Draw": 3.40, "Away": 3.60})
        assert sum(fair.values()) == pytest.approx(1.0, abs=1e-9)
        assert fair["Home"] > fair["Draw"] > fair["Away"]

    def test_devig_removes_the_margin(self):
        # A 1.90/1.90 book is 105.26%; fair is 50/50.
        fair = devig_proportional({"Over": 1.90, "Under": 1.90})
        assert fair["Over"] == pytest.approx(0.5, abs=1e-9)

    def test_devig_ignores_impossible_prices(self):
        fair = devig_proportional({"Home": 2.0, "Draw": 1.0, "Away": 0.0})
        assert set(fair) == {"Home"}

    def test_devig_empty_input(self):
        assert devig_proportional({}) == {}

    def test_overround_fair_book(self):
        assert overround({"Over": 2.0, "Under": 2.0}) == pytest.approx(0.0)

    def test_overround_typical_book(self):
        assert overround({"Over": 1.90, "Under": 1.90}) == pytest.approx(0.052632, abs=1e-6)

    def test_overround_no_prices(self):
        assert overround({"Over": 1.0}) is None


# ------------------------------------------------------------------
# Grouping + consensus
# ------------------------------------------------------------------

class TestMarketGroups:
    def test_groups_by_market_and_line(self):
        groups = collect_market_groups([
            book("A", "1X2", [("Home", 2.0), ("Draw", 3.4), ("Away", 3.6)]),
            book("A", "OU", [("Over", 1.9), ("Under", 1.95)], point=2.5),
            book("B", "OU", [("Over", 2.0), ("Under", 1.85)], point=3.5),
        ])
        assert set(groups) == {"1X2|", "OU|2.5", "OU|3.5"}
        assert groups["OU|2.5"].label == "OU 2.5"

    def test_over_under_without_a_line_is_dropped(self):
        # An OU quote with no line is not comparable to anything, so keeping it
        # would silently mix 2.5 and 3.5 goals into one "market".
        groups = collect_market_groups([book("A", "OU", [("Over", 1.9), ("Under", 1.9)])])
        assert groups == {}

    def test_handicap_away_line_is_normalised_to_home(self):
        groups = collect_market_groups([{
            "name": "A",
            "markets": [{"key": "HDP", "selections": [
                {"name": "Home", "odd": 1.95, "point": -0.5},
                {"name": "Away", "odd": 1.95, "point": 0.5},
            ]}],
        }])
        assert set(groups) == {"HDP|-0.5"}
        assert set(groups["HDP|-0.5"].books["A"]) == {"Home", "Away"}

    def test_best_price_and_spread_across_books(self):
        groups = collect_market_groups([
            book("A", "1X2", [("Home", 2.00), ("Draw", 3.4), ("Away", 3.6)]),
            book("B", "1X2", [("Home", 2.15), ("Draw", 3.3), ("Away", 3.5)]),
        ])
        group = groups["1X2|"]
        assert group.best_price("Home") == ("B", 2.15)
        assert group.price_spread("Home") == pytest.approx(0.15, abs=1e-9)
        assert group.book_count("Home") == 2

    def test_incomplete_books_are_excluded_from_consensus(self):
        groups = collect_market_groups([
            book("A", "1X2", [("Home", 2.0), ("Draw", 3.4), ("Away", 3.6)]),
            book("B", "1X2", [("Home", 2.5)]),  # only one outcome, unusable
        ])
        consensus = groups["1X2|"].consensus()
        assert consensus["books_used"] == 1
        # Probabilities are stored rounded to 6 dp, so the sum is 1.0 only to
        # within that rounding.
        assert sum(consensus["fair_probabilities"].values()) == pytest.approx(1.0, abs=1e-5)

    def test_consensus_uses_the_median(self):
        groups = collect_market_groups([
            book("A", "OU", [("Over", 1.80), ("Under", 2.00)], point=2.5),
            book("B", "OU", [("Over", 1.90), ("Under", 1.90)], point=2.5),
            book("C", "OU", [("Over", 2.00), ("Under", 1.80)], point=2.5),
        ])
        consensus = groups["OU|2.5"].consensus()
        assert consensus["books_used"] == 3
        # The middle book is fair 50/50, and the outliers sit either side of it.
        assert consensus["fair_probabilities"]["Over"] == pytest.approx(0.5, abs=1e-6)
        assert consensus["dispersion"]["Over"] > 0
        assert consensus["average_overround"] > 0


# ------------------------------------------------------------------
# Candidates: line shopping is the edge
# ------------------------------------------------------------------

def three_book_1x2() -> dict:
    """Three books on 1X2 where book C is clearly the best price on Home."""
    return collect_market_groups([
        book("A", "1X2", [("Home", 2.00), ("Draw", 3.40), ("Away", 3.80)]),
        book("B", "1X2", [("Home", 2.05), ("Draw", 3.35), ("Away", 3.70)]),
        book("C", "1X2", [("Home", 2.30), ("Draw", 3.30), ("Away", 3.60)]),
    ])


class TestCandidates:
    def test_edge_comes_from_the_best_price(self):
        candidates = build_candidates(three_book_1x2(), min_books=3)
        home = next(c for c in candidates if c.selection == "Home")
        assert home.bookmaker == "C"
        assert home.odds == 2.30
        # Consensus says Home is worth ~2.05, so 2.30 is real value.
        assert home.edge > 0
        assert home.implied_probability == pytest.approx(1 / 2.30, abs=1e-9)
        assert home.edge == pytest.approx(
            home.blended_probability - home.implied_probability, abs=1e-9)
        assert home.ev == pytest.approx(
            home.blended_probability * (home.odds - 1) - (1 - home.blended_probability),
            abs=1e-9)

    def test_thin_markets_are_skipped(self):
        # Two books cannot establish a consensus worth betting into.
        groups = collect_market_groups([
            book("A", "1X2", [("Home", 2.0), ("Draw", 3.4), ("Away", 3.6)]),
            book("B", "1X2", [("Home", 2.1), ("Draw", 3.3), ("Away", 3.5)]),
        ])
        assert build_candidates(groups, min_books=3) == []
        assert build_candidates(groups, min_books=2) != []

    def test_sorted_by_expected_value(self):
        candidates = build_candidates(three_book_1x2(), min_books=3)
        evs = [c.ev for c in candidates]
        assert evs == sorted(evs, reverse=True)

    def test_model_probability_is_blended_not_trusted_blindly(self):
        groups = three_book_1x2()
        consensus_only = build_candidates(groups, min_books=3)
        base = next(c for c in consensus_only if c.selection == "Home")
        with_model = build_candidates(
            groups, model_probabilities={"1X2|Home": 0.70}, model_weight=0.35,
            min_books=3)
        blended = next(c for c in with_model if c.selection == "Home")
        expected = 0.65 * base.consensus_probability + 0.35 * 0.70
        assert blended.blended_probability == pytest.approx(expected, abs=1e-6)
        assert blended.model_probability == 0.70

    def test_candidate_keys_include_the_line(self):
        groups = collect_market_groups([
            book("A", "OU", [("Over", 1.90), ("Under", 1.95)], point=2.5),
            book("B", "OU", [("Over", 1.95), ("Under", 1.90)], point=2.5),
            book("C", "OU", [("Over", 2.05), ("Under", 1.85)], point=2.5),
        ])
        candidates = build_candidates(
            groups, model_probabilities={"OU|2.5|Over": 0.60}, model_weight=1.0,
            min_books=3)
        over = next(c for c in candidates if c.selection == "Over")
        assert over.point == 2.5
        assert over.model_probability == 0.60
        assert over.blended_probability == pytest.approx(0.60, abs=1e-9)
        assert over.label == "OU 2.5 — Over"

    def test_to_dict_is_json_friendly(self):
        candidate = build_candidates(three_book_1x2(), min_books=3)[0]
        payload = candidate.to_dict()
        assert payload["market"] == "1X2"
        assert payload["model_probability"] is None
        assert isinstance(payload["book_count"], int)


class TestBlendAndTotals:
    def test_blend_without_model_returns_consensus(self):
        assert blend_probability(0.42, None, 0.35) == 0.42

    def test_blend_weight_is_clamped(self):
        assert blend_probability(0.4, 0.8, 5.0) == pytest.approx(0.8)
        assert blend_probability(0.4, 0.8, -1.0) == pytest.approx(0.4)

    def test_totals_estimate_prefers_the_most_liquid_line(self):
        groups = collect_market_groups([
            # 3.5 line quoted by one book only.
            book("A", "OU", [("Over", 2.60), ("Under", 1.50)], point=3.5),
            # 2.5 line quoted by three books — this is the liquid one.
            book("A2", "OU", [("Over", 1.90), ("Under", 1.95)], point=2.5),
            book("B", "OU", [("Over", 1.95), ("Under", 1.90)], point=2.5),
            book("C", "OU", [("Over", 1.92), ("Under", 1.92)], point=2.5),
        ])
        total = market_totals_estimate(groups, default_total=2.7)
        assert total is not None
        assert 2.4 < total < 3.1

    def test_totals_estimate_without_ou_data(self):
        groups = collect_market_groups([
            book("A", "1X2", [("Home", 2.0), ("Draw", 3.4), ("Away", 3.6)]),
        ])
        assert market_totals_estimate(groups, default_total=2.7) is None
