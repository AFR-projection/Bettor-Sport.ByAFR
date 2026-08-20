"""Market mathematics for AI Bettor.

This is the quantitative core that replaced the old hardcoded assumptions
(fixed 1.5/1.2 goal expectancies and a fixed 1.91 price). Everything here is
derived from the odds that Data Scout actually fetched.

Pipeline of ideas:

1. **Per-book de-vig** — a bookmaker's prices contain an overround. Removing it
   proportionally turns prices into that book's implied fair probabilities.
2. **Cross-book consensus** — the median fair probability across books is a
   robust estimate of the market's true price ("fair probability").
3. **Poisson calibration** — the consensus 1X2 probabilities plus the Over/Under
   line give expected goals for each side (lambda), so the Monte Carlo engine
   simulates the *actual* match instead of a generic one.
4. **Value detection** — a bet has value when the best price available at any
   book implies a probability lower than the consensus fair probability.
   That is real line shopping, not a guess.

No function here invents a price: if the market data is insufficient, the
functions return None / empty and the caller must treat it as NO BET.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Number of outcomes a market must expose before its overround can be removed.
EXPECTED_OUTCOMES = {"1X2": 3, "OU": 2, "HDP": 2}

MAX_GOALS_GRID = 12


# ----------------------------------------------------------------------
# Poisson helpers (exact, no sampling)
# ----------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam), computed in log space."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_1x2(home_lambda: float, away_lambda: float, max_goals: int = MAX_GOALS_GRID) -> Tuple[float, float, float]:
    """Exact 1X2 probabilities for independent Poisson scorelines."""
    home_win = draw = away_win = 0.0
    home_pmf = [poisson_pmf(i, home_lambda) for i in range(max_goals + 1)]
    away_pmf = [poisson_pmf(i, away_lambda) for i in range(max_goals + 1)]
    for h, ph in enumerate(home_pmf):
        for a, pa in enumerate(away_pmf):
            p = ph * pa
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p
    total = home_win + draw + away_win
    if total <= 0:
        return 0.0, 0.0, 0.0
    return home_win / total, draw / total, away_win / total


def poisson_over(home_lambda: float, away_lambda: float, line: float,
                 max_goals: int = MAX_GOALS_GRID) -> float:
    """P(total goals > line). Total goals of two Poissons is Poisson(sum)."""
    total_lambda = max(0.01, home_lambda + away_lambda)
    under_or_equal = 0.0
    for goals in range(0, max_goals * 2 + 1):
        if goals <= line:
            under_or_equal += poisson_pmf(goals, total_lambda)
    return max(0.0, min(1.0, 1.0 - under_or_equal))


def poisson_handicap_cover(home_lambda: float, away_lambda: float, line: float,
                           max_goals: int = MAX_GOALS_GRID) -> float:
    """P(home margin + line > 0) — Asian handicap cover probability for home."""
    cover = push = 0.0
    home_pmf = [poisson_pmf(i, home_lambda) for i in range(max_goals + 1)]
    away_pmf = [poisson_pmf(i, away_lambda) for i in range(max_goals + 1)]
    for h, ph in enumerate(home_pmf):
        for a, pa in enumerate(away_pmf):
            adjusted = (h - a) + line
            if adjusted > 1e-9:
                cover += ph * pa
            elif abs(adjusted) <= 1e-9:
                push += ph * pa
    remaining = cover + (1.0 - cover - push)
    if remaining <= 0:
        return 0.0
    # Pushes return the stake, so they are excluded from the win probability.
    return cover / (1.0 - push) if push < 1 else 0.0


def solve_total_goals(over_probability: float, line: float,
                      low: float = 0.4, high: float = 7.0) -> Optional[float]:
    """Find total expected goals whose Poisson P(over line) matches the market."""
    if not (0.01 < over_probability < 0.99):
        return None
    for _ in range(60):
        mid = (low + high) / 2
        p_over = poisson_over(mid / 2, mid / 2, line)
        if p_over < over_probability:
            low = mid
        else:
            high = mid
    return round((low + high) / 2, 4)


def solve_supremacy(home_probability: float, total_goals: float) -> float:
    """Find the goal supremacy that reproduces the market's home win chance."""
    low, high = -total_goals * 0.95, total_goals * 0.95
    for _ in range(60):
        mid = (low + high) / 2
        home_lambda = max(0.05, (total_goals + mid) / 2)
        away_lambda = max(0.05, (total_goals - mid) / 2)
        p_home, _, _ = poisson_1x2(home_lambda, away_lambda)
        if p_home < home_probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def fit_lambdas(
    home_probability: Optional[float],
    away_probability: Optional[float],
    total_goals: Optional[float],
    default_total: float = 2.7,
) -> Tuple[float, float]:
    """Derive (home_lambda, away_lambda) from market probabilities.

    Falls back to a symmetric split of `default_total` when the market gives us
    nothing usable, so the simulation still runs but carries no false signal.
    """
    total = total_goals if (total_goals and total_goals > 0.4) else default_total
    total = max(0.6, min(7.0, total))
    if home_probability is None or not (0.01 < home_probability < 0.99):
        return round(total / 2, 4), round(total / 2, 4)
    supremacy = solve_supremacy(home_probability, total)
    home_lambda = max(0.05, (total + supremacy) / 2)
    away_lambda = max(0.05, (total - supremacy) / 2)
    return round(home_lambda, 4), round(away_lambda, 4)


# ----------------------------------------------------------------------
# De-vig / consensus
# ----------------------------------------------------------------------

def devig_proportional(odds_by_selection: Dict[str, float]) -> Dict[str, float]:
    """Remove a book's overround proportionally (a.k.a. multiplicative method)."""
    implied = {
        name: 1.0 / odd
        for name, odd in odds_by_selection.items()
        if isinstance(odd, (int, float)) and odd > 1
    }
    total = sum(implied.values())
    if total <= 0:
        return {}
    return {name: value / total for name, value in implied.items()}


def overround(odds_by_selection: Dict[str, float]) -> Optional[float]:
    """Bookmaker margin, e.g. 0.05 == 105% book."""
    implied = [1.0 / odd for odd in odds_by_selection.values()
               if isinstance(odd, (int, float)) and odd > 1]
    if not implied:
        return None
    return round(sum(implied) - 1.0, 6)


@dataclass
class MarketGroup:
    """All prices for one market + line across every bookmaker."""

    market: str
    point: Optional[float] = None
    # selection -> [(bookmaker, odds)]
    prices: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    # bookmaker -> {selection: odds} (only complete books)
    books: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.point is None:
            return self.market
        return f"{self.market} {self.point:+g}" if self.market == "HDP" else f"{self.market} {self.point:g}"

    def best_price(self, selection: str) -> Optional[Tuple[str, float]]:
        quotes = self.prices.get(selection) or []
        if not quotes:
            return None
        book, odd = max(quotes, key=lambda q: q[1])
        return book, odd

    def price_spread(self, selection: str) -> float:
        quotes = [odd for _, odd in self.prices.get(selection) or []]
        if len(quotes) < 2:
            return 0.0
        return round(max(quotes) - min(quotes), 4)

    def book_count(self, selection: str) -> int:
        return len(self.prices.get(selection) or [])

    def complete_books(self) -> Dict[str, Dict[str, float]]:
        """Books quoting every outcome, so their overround can be removed."""
        expected = EXPECTED_OUTCOMES.get(self.market, 2)
        return {
            book: quotes for book, quotes in self.books.items()
            if len(quotes) >= expected
        }

    def consensus(self) -> Dict[str, Any]:
        """Median de-vigged probability per selection across complete books."""
        complete = self.complete_books()
        per_selection: Dict[str, List[float]] = {}
        margins: List[float] = []
        for quotes in complete.values():
            fair = devig_proportional(quotes)
            if not fair:
                continue
            margin = overround(quotes)
            if margin is not None:
                margins.append(margin)
            for selection, probability in fair.items():
                per_selection.setdefault(selection, []).append(probability)

        fair_probs: Dict[str, float] = {}
        dispersion: Dict[str, float] = {}
        for selection, values in per_selection.items():
            fair_probs[selection] = round(statistics.median(values), 6)
            dispersion[selection] = round(
                statistics.pstdev(values) if len(values) > 1 else 0.0, 6
            )

        total = sum(fair_probs.values())
        if total > 0:
            fair_probs = {k: round(v / total, 6) for k, v in fair_probs.items()}

        return {
            "fair_probabilities": fair_probs,
            "dispersion": dispersion,
            "books_used": len(complete),
            "average_overround": round(sum(margins) / len(margins), 6) if margins else None,
        }


def collect_market_groups(odds_data: Iterable[Dict[str, Any]]) -> Dict[str, MarketGroup]:
    """Group every quote in the canonical odds structure by market + line."""
    groups: Dict[str, MarketGroup] = {}
    for book in odds_data or []:
        if not isinstance(book, dict):
            continue
        book_name = str(book.get("name") or book.get("key") or "UNKNOWN")
        for market in book.get("markets") or []:
            if not isinstance(market, dict):
                continue
            market_key = str(market.get("key") or "")
            if not market_key:
                continue
            for selection in market.get("selections") or []:
                if not isinstance(selection, dict):
                    continue
                odd = selection.get("odd")
                if not isinstance(odd, (int, float)) or odd <= 1:
                    continue
                point = selection.get("point", market.get("point"))
                try:
                    point_value = float(point) if point is not None else None
                except (TypeError, ValueError):
                    point_value = None
                # Over/Under and handicap lines are only comparable at the same
                # line, so the line is part of the group identity.
                if market_key in ("OU", "HDP") and point_value is None:
                    continue
                if market_key == "HDP":
                    # Normalise so the group key is the home line.
                    if str(selection.get("name")) == "Away" and point_value is not None:
                        point_value = -point_value
                group_key = f"{market_key}|{point_value if point_value is not None else ''}"
                group = groups.get(group_key)
                if group is None:
                    group = MarketGroup(market=market_key, point=point_value)
                    groups[group_key] = group
                name = str(selection.get("name") or "UNKNOWN")
                group.prices.setdefault(name, []).append((book_name, float(odd)))
                book_quotes = group.books.setdefault(book_name, {})
                # Keep the best price this book offers for the selection.
                if float(odd) > book_quotes.get(name, 0.0):
                    book_quotes[name] = float(odd)
    return groups


# ----------------------------------------------------------------------
# Candidate generation
# ----------------------------------------------------------------------

@dataclass
class Candidate:
    """One concrete betting opportunity with its price and probabilities."""

    market: str
    selection: str
    point: Optional[float]
    odds: float
    bookmaker: str
    consensus_probability: float
    model_probability: Optional[float]
    blended_probability: float
    implied_probability: float
    edge: float
    ev: float
    book_count: int
    price_spread: float
    dispersion: float
    average_overround: Optional[float]
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market": self.market,
            "selection": self.selection,
            "point": self.point,
            "odds": round(self.odds, 4),
            "bookmaker": self.bookmaker,
            "consensus_probability": round(self.consensus_probability, 6),
            "model_probability": round(self.model_probability, 6) if self.model_probability is not None else None,
            "blended_probability": round(self.blended_probability, 6),
            "implied_probability": round(self.implied_probability, 6),
            "edge": round(self.edge, 6),
            "ev": round(self.ev, 6),
            "book_count": self.book_count,
            "price_spread": self.price_spread,
            "dispersion": self.dispersion,
            "average_overround": self.average_overround,
            "label": self.label,
        }


def market_totals_estimate(groups: Dict[str, MarketGroup], default_total: float) -> Optional[float]:
    """Expected total goals implied by the most liquid Over/Under line."""
    best: Optional[Tuple[int, float, float]] = None  # (books, line, p_over)
    for group in groups.values():
        if group.market != "OU" or group.point is None:
            continue
        consensus = group.consensus()
        probabilities = consensus["fair_probabilities"]
        p_over = probabilities.get("Over")
        if p_over is None:
            continue
        books = consensus["books_used"]
        if best is None or books > best[0]:
            best = (books, group.point, p_over)
    if best is None:
        return None
    _, line, p_over = best
    return solve_total_goals(p_over, line)


def blend_probability(consensus: float, model: Optional[float], model_weight: float) -> float:
    """Weighted blend of market consensus and the simulation model."""
    if model is None:
        return consensus
    weight = max(0.0, min(1.0, model_weight))
    return (1.0 - weight) * consensus + weight * model


def build_candidates(
    groups: Dict[str, MarketGroup],
    model_probabilities: Optional[Dict[str, float]] = None,
    model_weight: float = 0.35,
    min_books: int = 3,
) -> List[Candidate]:
    """Create one candidate per (market, line, selection) with a fair price.

    `model_probabilities` maps a candidate key ("1X2|Home", "OU|2.5|Over",
    "HDP|-0.5|Home") to the Monte Carlo probability for that outcome.
    """
    model_probabilities = model_probabilities or {}
    candidates: List[Candidate] = []

    for group in groups.values():
        consensus = group.consensus()
        fair_probabilities = consensus["fair_probabilities"]
        if not fair_probabilities:
            continue
        for selection, fair_probability in fair_probabilities.items():
            if not (0.0 < fair_probability < 1.0):
                continue
            best = group.best_price(selection)
            if best is None:
                continue
            bookmaker, odds = best
            book_count = group.book_count(selection)
            if book_count < max(1, min_books):
                continue

            key_parts = [group.market]
            if group.point is not None:
                key_parts.append(f"{group.point:g}")
            key_parts.append(selection)
            model_probability = model_probabilities.get("|".join(key_parts))

            blended = blend_probability(fair_probability, model_probability, model_weight)
            blended = max(0.005, min(0.995, blended))
            implied = 1.0 / odds
            edge = blended - implied
            ev = blended * (odds - 1.0) - (1.0 - blended)

            candidates.append(Candidate(
                market=group.market,
                selection=selection,
                point=group.point,
                odds=odds,
                bookmaker=bookmaker,
                consensus_probability=fair_probability,
                model_probability=model_probability,
                blended_probability=blended,
                implied_probability=implied,
                edge=edge,
                ev=ev,
                book_count=book_count,
                price_spread=group.price_spread(selection),
                dispersion=consensus["dispersion"].get(selection, 0.0),
                average_overround=consensus["average_overround"],
                label=f"{group.label} — {selection}",
            ))

    candidates.sort(key=lambda c: c.ev, reverse=True)
    return candidates
