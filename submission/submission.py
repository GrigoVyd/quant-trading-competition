from __future__ import annotations

from typing import Dict, List, Any
import logging

# --- Participant Imports ---
# These modules are mocked by the evaluator_lambda.py and will only
# be available when running inside the Lambda environment.
try:
    from pricing.Market import Market
    from pricing.Portfolio import Portfolio
except ImportError:
    print("Failed to import backtest modules. Are you running locally?")
    # Define dummy classes for local linting/type-checking if needed
    class Market: pass
    class Portfolio: pass

# --- Setup Logger ---
logger = logging.getLogger("local_eval")

# --- Define the Trader ---
class ReversalTrader:
    """
    Implements a percentage-based reversal strategy for all products
    in the market universe.

    Lever Logic:
    - BUY: When price goes to the lowest recorded value (LWM) and then tips up by 8%. (Increased)
    - SELL: When price reaches the highest recorded value (HWM) and then dips by 4%. (Increased)
    """

    # --- Strategy Configuration ---
    BUY_REBOUND_PERCENT = 1.07  # 7% rebound from LWM
    SELL_DIP_PERCENT = 0.93     # 7% dip from HWM
    TRADE_SIZE = 10000          # Fixed quantity for each trade

    def __init__(self) -> None:
        logger.debug("ReversalTrader initialized.")
        
        # State to track the highest and lowest price seen for each product
        self.high_water_mark: Dict[str, float] = {}
        self.low_water_mark: Dict[str, float] = {}

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
        # 1. Multi-Product Extraction: Get all products in the market
        universe: List[str] = list(quotes.keys())

        for product in universe:
            
            price = self._get_current_price(quotes.get(product))
            if price is None:
                continue

            # Initialize or update water marks
            if product not in self.high_water_mark:
                # Initialize LWM/HWM on the first tick for this product
                self.high_water_mark[product] = price
                self.low_water_mark[product] = price
                continue
            
            # --- Update Water Marks ---
            
            # HWM tracks the highest price ever seen
            self.high_water_mark[product] = max(self.high_water_mark[product], price)
            
            # LWM tracks the lowest price ever seen
            self.low_water_mark[product] = min(self.low_water_mark[product], price)
            
            current_position = portfolio.positions.get(product, 0)
            
            # --- Buy Logic (8% Rebound from Low Water Mark) ---
            
            # If the current price is 8% above the lowest price recorded
            buy_threshold = self.low_water_mark[product] * self.BUY_REBOUND_PERCENT
            
            # We buy if we don't have a large long position (or we have a short position)
            # The 'current_position <= self.TRADE_SIZE / 2' is for position management (explained below)
            if current_position <= self.TRADE_SIZE / 2 and price >= buy_threshold:
                # Execute buy
                portfolio.buy(product, self.TRADE_SIZE)
                # Reset the LWM to the current price to start tracking the *next* low point
                self.low_water_mark[product] = price
                logger.info(f"BUY {product}: Price {price:.4f} > Rebound Threshold {buy_threshold:.4f}")
                
            # --- Sell Logic (4% Dip from High Water Mark) ---
            
            # If the current price is 4% below the highest price recorded
            sell_threshold = self.high_water_mark[product] * self.SELL_DIP_PERCENT
            
            # We sell if we don't have a large short position (or we have a long position)
            # The 'current_position >= -self.TRADE_SIZE / 2' is for position management (explained below)
            if current_position >= -self.TRADE_SIZE / 2 and price <= sell_threshold:
                # Execute sell
                portfolio.sell(product, self.TRADE_SIZE)
                # Reset the HWM to the current price to start tracking the *next* high point
                self.high_water_mark[product] = price
                logger.info(f"SELL {product}: Price {price:.4f} < Dip Threshold {sell_threshold:.4f}")


# --- Define the Factory Function ---
def build_trader() -> ReversalTrader:
    """
    Factory function to build and return the trader instance.
    """
    return ReversalTrader()