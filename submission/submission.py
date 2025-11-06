from __future__ import annotations

import math
from collections import deque
from typing import Dict, List

import numpy as np

# Pricing interfaces are provided by the evaluator at runtime. Provide
# light stubs for local linting / module import safety.
try:
    from pricing.Market import Market
    from pricing.Portfolio import Portfolio
except Exception:
    class Market: pass
    class Portfolio: pass


# -------------------------
# Configuration (tuneable)
# -------------------------
MU_ALPHA = 0.033
VAR_BETA = 0.011
VAR_FLOOR = 1e-6
PRICE_BUFFER_SIZE = 90
REBALANCE_EVERY = 5
SMOOTHING_GAMMA = 0.25
MIN_POS_DOLLARS = 1.0
MIN_TRADE_QTY = 1
MAX_WEIGHT_PER_ASSET = 0.45
MAX_GROSS_LEVERAGE = 1.0
MAX_UNIVERSE = None
TOP_K = None


class EWMAOnlineTrader:
    """Lightweight online EWMA trader suitable for Lambda constraints.

    - Keeps per-symbol EWMA mean and variance estimates.
    - Maps mu/var -> long-only weights using mu/var and enforces risk caps.
    - Applies smoothing and hysteresis to reduce turnover.
    """

    def __init__(self) -> None:
        # state per symbol
        self.price_buf: Dict[str, deque] = {}
        self.mu: Dict[str, float] = {}
        self.var: Dict[str, float] = {}
        self.prev_weights: Dict[str, float] = {}
        self.prev_qty: Dict[str, int] = {}
        self.call_count = 0

    # --- internal helpers ---
    def _update_price_and_stats(self, symbol: str, price: float) -> None:
        buf = self.price_buf.get(symbol)
        if buf is None:
            buf = deque(maxlen=PRICE_BUFFER_SIZE)
            self.price_buf[symbol] = buf
        # append new price
        if len(buf) > 0 and price == buf[-1]:
            # identical price; still keep but do not update stats to avoid zero returns
            buf.append(price)
            return
        prev = buf[-1] if len(buf) > 0 else None
        buf.append(price)
        if prev is None:
            return
        # compute simple return
        if prev == 0:
            return
        r = (price / prev) - 1.0
        # initialize if needed
        if symbol not in self.mu:
            self.mu[symbol] = float(r)
            self.var[symbol] = float(0.0)
            return
        # EWMA updates
        prev_mu = self.mu[symbol]
        new_mu = (1 - MU_ALPHA) * prev_mu + MU_ALPHA * r
        # variance update uses deviation from previous mean (more stable)
        prev_var = self.var[symbol]
        new_var = (1 - VAR_BETA) * prev_var + VAR_BETA * ((r - prev_mu) ** 2)
        self.mu[symbol] = float(new_mu)
        self.var[symbol] = float(new_var)

    def _collect_universe(self, market: Market) -> List[str]:
        universe = getattr(market, "universe", None)
        if universe is None:
            # market.quotes may be a dict mapping symbol->{...} or symbol->price
            quotes = getattr(market, "quotes", {})
            universe = list(quotes.keys())
        if MAX_UNIVERSE is not None and len(universe) > MAX_UNIVERSE:
            return list(universe)[:MAX_UNIVERSE]
        return list(universe)

    def _compute_weights(self, syms: List[str], nav: float) -> Dict[str, float]:
        # build arrays (only symbols with price and mu/var)
        prices = []
        mus = []
        vars_ = []
        good_syms = []
        for s in syms:
            buf = self.price_buf.get(s)
            if buf is None or len(buf) == 0:
                continue
            price = float(buf[-1])
            if math.isnan(price) or price <= 0:
                continue
            if s not in self.mu:
                continue
            v = float(self.var.get(s, 0.0))
            mus.append(float(self.mu.get(s, 0.0)))
            vars_.append(max(v, VAR_FLOOR))
            prices.append(price)
            good_syms.append(s)

        if len(good_syms) == 0:
            return {}

        mus = np.array(mus, dtype=float)
        vars_arr = np.array(vars_, dtype=float)

        # raw score: expected return per unit variance
        raw = mus / vars_arr
        # long-only: remove negatives
        raw = np.clip(raw, 0.0, None)

        total = float(raw.sum())
        if total <= 0:
            # fallback: equal weight among assets with non-negative mu or among all good_syms
            pos_mask = mus > 0
            if pos_mask.sum() > 0:
                w = pos_mask.astype(float) / float(pos_mask.sum())
            else:
                w = np.ones(len(good_syms), dtype=float) / float(len(good_syms))
        else:
            w = raw / total

        # enforce per-asset cap
        if MAX_WEIGHT_PER_ASSET is not None:
            w = np.minimum(w, MAX_WEIGHT_PER_ASSET)
            s = float(w.sum())
            if s <= 0:
                # degenerate after capping -> equal weight fallback
                w = np.ones_like(w) / float(len(w))
            else:
                w = w / s

        # smoothing with previous weights (align previous weights to current good_syms)
        prev_w = np.array([self.prev_weights.get(s, 0.0) for s in good_syms], dtype=float)
        w = (1 - SMOOTHING_GAMMA) * prev_w + SMOOTHING_GAMMA * w

        # enforce gross leverage cap (for long-only weights sum to 1 so gross==1; but keep generic)
        gross = float(np.sum(np.abs(w)))
        if gross > MAX_GROSS_LEVERAGE and gross > 0:
            w = w * (MAX_GROSS_LEVERAGE / gross)

        # prepare mapping symbol->weight
        return {s: float(wi) for s, wi in zip(good_syms, w)}

    def _weights_to_target_qty(self, weights: Dict[str, float], portfolio: Portfolio, nav: float) -> Dict[str, int]:
        targets: Dict[str, int] = {}
        # convert weights->dollar allocations->integer shares
        for s, w in weights.items():
            price = float(self.price_buf[s][-1])
            alloc = w * nav
            if alloc < MIN_POS_DOLLARS or price <= 0:
                qty = 0
            else:
                qty = int(round(alloc / price))
            targets[s] = qty
        return targets

    def _execute_trades(self, targets: Dict[str, int], portfolio: Portfolio) -> None:
        cur_positions = getattr(portfolio, "positions", {}) or {}
        for s, tgt in targets.items():
            cur = int(cur_positions.get(s, 0))
            diff = tgt - cur
            if abs(diff) < MIN_TRADE_QTY:
                continue
            if diff > 0:
                portfolio.buy(product=s, quantity=int(diff))
                self.prev_qty[s] = tgt
            elif diff < 0:
                portfolio.sell(product=s, quantity=int(abs(diff)))
                self.prev_qty[s] = tgt

    # --- public interface ---
    def on_quote(self, market: Market, portfolio: Portfolio) -> None:
        # Update internal state with incoming quotes
        self.call_count += 1
        quotes = getattr(market, "quotes", {})
        universe = self._collect_universe(market)

        for s in universe:
            q = quotes.get(s)
            if q is None:
                continue
            # quote may be a dict with 'price' or a simple numeric
            price = q["price"] if isinstance(q, dict) and "price" in q else q
            try:
                p = float(price)
            except Exception:
                continue
            if math.isnan(p):
                continue
            self._update_price_and_stats(s, p)

        # Only rebalance every REBALANCE_EVERY calls
        if (self.call_count % REBALANCE_EVERY) != 0:
            return

        # compute NAV using portfolio helper if available
        nav = 1.0
        nav_fn = getattr(portfolio, "_net_asset_value", None)
        if callable(nav_fn):
            try:
                nav = float(nav_fn())
                if nav <= 0 or math.isnan(nav):
                    nav = 1.0
            except Exception:
                nav = 1.0

        # compute weights and targets
        weights = self._compute_weights(universe, nav)
        if not weights:
            return
        targets = self._weights_to_target_qty(weights, portfolio, nav)

        # execute trades with hysteresis
        self._execute_trades(targets, portfolio)

        # update prev_weights for symbols considered
        for s, w in weights.items():
            self.prev_weights[s] = w


def build_trader() -> EWMAOnlineTrader:
    return EWMAOnlineTrader()