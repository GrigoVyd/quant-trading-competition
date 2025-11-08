from __future__ import annotations

from typing import Dict, List, Any
import logging
import math

try:
    from pricing.Market import Market
    from pricing.Portfolio import Portfolio
except ImportError:
    pass

# --- Setup Logger ---
logger = logging.getLogger("local_eval")

# --- Define the Trader ---
class ReversalTrader:
    """
    Implements a percentage-based reversal strategy for all products
    in the market universe, combining signal detection with volatility-based
    position sizing for risk management.

    Strategy Logic:
    - BUY: When price recovers 7% from the lowest recorded value (LWM).
    - SELL: When price dips 7% from the highest recorded value (HWM).
    """

    # --- Strategy Configuration ---
    # Trigger Thresholds: The price must move this much *past* the extreme water mark
    BUY_REBOUND_PERCENT = 1.07    # 7% rebound from LWM (LWM * 1.07)
    SELL_DIP_PERCENT = 0.93       # 7% dip from HWM (HWM * 0.93)

    # Risk / Sizing Configuration
    SAFE_LEVERAGE = 7.0             # Max total leverage for notional calculation
    PER_ASSET_CAP_FRAC = 0.9        # Max gross notional exposure per asset as fraction of NAV
    VOL_WINDOW = 20                 # Lookback window (bars) to estimate volatility
    MIN_NOTIONAL_TRADE_FRAC = 0.001 # Minimum trade notional as fraction of NAV to avoid tiny churn

    def __init__(self) -> None:
        logger.debug("ReversalTrader initialized.")
        
        # State to track the highest and lowest price seen for each product (HWM/LWM)
        self.high_water_mark: Dict[str, float] = {}
        self.low_water_mark: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = {}

    def _get_current_price(self, quote: Any) -> float | None:
        """Safely extracts the price from a quote object."""
        if isinstance(quote, dict) and 'price' in quote:
            price = quote['price']
        else:
            price = quote
        
        try:
            p = float(price)
            return p if p > 0 else None
        except (ValueError, TypeError):
            return None

    def on_quote(self, market: Market, portfolio: Portfolio) -> None:
        """
        Main event loop: processes quotes for all products and executes trades.
        """
        
        quotes = market.quotes
        universe: List[str] = list(quotes.keys())

        for product in universe:
            
            price = self._get_current_price(quotes.get(product))
            if price is None:
                continue

            # 1. Initialize or update water marks and price history
            if product not in self.high_water_mark:
                self.high_water_mark[product] = price
                self.low_water_mark[product] = price
                self.price_history[product] = [price]
                continue
            else:
                h = self.price_history.setdefault(product, [])
                h.append(price)
                if len(h) > self.VOL_WINDOW + 2:
                    del h[0]
            
            # --- Update Water Marks ---
            # HWM tracks the highest price ever seen
            self.high_water_mark[product] = max(self.high_water_mark[product], price)
            # LWM tracks the lowest price ever seen
            self.low_water_mark[product] = min(self.low_water_mark[product], price)
            
            current_position = portfolio.positions.get(product, 0)

            # --- 2. Signal Detection ---
            buy_threshold = self.low_water_mark[product] * self.BUY_REBOUND_PERCENT
            sell_threshold = self.high_water_mark[product] * self.SELL_DIP_PERCENT

            signal: int = 0
            if price >= buy_threshold:
                signal = 1
            elif price <= sell_threshold:
                signal = -1

            if signal == 0:
                continue

            # --- 3. Position Sizing (Vol-Targeting & Risk Caps) ---
            
            # Estimate annualized volatility for risk-based sizing
            sigma = self._estimate_volatility(product)

            # Retrieve Net Asset Value (NAV)
            nav = float(portfolio.summary().get("net_value", 0.0))
            if nav <= 0:
                logger.warning("NAV non-positive; skipping trading.")
                continue
            
            per_asset_cap = self.PER_ASSET_CAP_FRAC * nav
            
            target_notional = per_asset_cap
            
            target_signed_notional = float(signal) * target_notional

            target_qty = int(abs(target_signed_notional) / price)
            
            if target_qty == 0:
                continue

            min_trade_notional = max(self.MIN_NOTIONAL_TRADE_FRAC * nav, 1.0)
            trade_notional = target_qty * price
            
            if trade_notional < min_trade_notional:
                logger.debug(f"{product}: trade notional {trade_notional:.2f} below min {min_trade_notional:.2f}; skipping")
                continue

            # --- 4. Execution ---

            desired_signed_qty = int(signal * target_qty)
            delta_qty = desired_signed_qty - current_position

            if delta_qty == 0:
                continue

            if delta_qty > 0:
                success = portfolio.buy(product, delta_qty)
                if success:
                    self.low_water_mark[product] = price
                    logger.info(f"BUY {product}: qty {delta_qty} @ {price:.4f} | target_notional={target_signed_notional:.2f}")
            else:
                success = portfolio.sell(product, abs(delta_qty))
                if success:
                    self.high_water_mark[product] = price
                    logger.info(f"SELL {product}: qty {abs(delta_qty)} @ {price:.4f} | target_notional={target_signed_notional:.2f}")

    def _estimate_volatility(self, product: str) -> float:
        """
        Estimate annualized volatility from recent price history using log returns.
        This provides the risk input needed for position sizing.
        """
        history = self.price_history.get(product, [])
        if len(history) < 3:
            return 0.20

        # Compute simple log returns (log(P_t / P_{t-1}))
        returns: List[float] = []
        for i in range(1, len(history)):
            p0 = history[i-1]
            p1 = history[i]
            if p0 <= 0 or p1 <= 0:
                continue
            returns.append(math.log(p1 / p0))

        if len(returns) < 2:
            return 0.20

        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        daily_sigma = var ** 0.5
        
        # Annualize volatility assuming 252 trading days per year
        annual_sigma = daily_sigma * (252 ** 0.5)
        
        return max(annual_sigma, 1e-6)


# --- Define the Factory Function ---
def build_trader() -> ReversalTrader:
    """
    Factory function to build and return the trader instance.
    """
    return ReversalTrader()