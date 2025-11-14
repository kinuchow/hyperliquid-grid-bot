#!/usr/bin/env python3
import json
import os
import time
import numpy as np
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple, Callable

# Configure logging if not already configured
logger = logging.getLogger(__name__)

_ENV_FILE_CACHE: Optional[Dict[str, str]] = None

def _load_env_file(path: str) -> Dict[str, str]:
    """Load KEY=VALUE pairs from a .env file if it exists."""
    env_vars: Dict[str, str] = {}
    try:
        if not os.path.exists(path):
            return env_vars
        with open(path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    env_vars[key] = value
    except Exception as exc:
        logger.warning("Unable to parse env file %s: %s", path, exc)
    return env_vars

def _get_env_value(key: str) -> Optional[str]:
    """Retrieve value for key from environment or .env file cache."""
    value = os.environ.get(key)
    if value is not None:
        return value
    global _ENV_FILE_CACHE
    if _ENV_FILE_CACHE is None:
        env_file_path = os.environ.get("GRID_ENV_FILE", ".env")
        _ENV_FILE_CACHE = _load_env_file(env_file_path)
    return _ENV_FILE_CACHE.get(key) if _ENV_FILE_CACHE else None

def _env_or_default(key: str, default: Any, caster: Callable[[str], Any]) -> Any:
    """
    Read an environment variable, cast it, and fall back to a default if unset/invalid.
    """
    value = _get_env_value(key)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    try:
        return caster(stripped)
    except (ValueError, TypeError):
        logger.warning("Invalid value for %s=%s. Using default %s", key, value, default)
        return default

def _round_value_to_tick(value: float, tick: Optional[float]) -> float:
    """Round a numeric value to the configured tick size."""
    if not tick:
        return round(value, 6)
    tick_dec = Decimal(str(tick))
    scaled = (Decimal(str(value)) / tick_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    rounded = scaled * tick_dec
    return float(rounded)

# Import hyperliquid packages
try:
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    logger.info("Successfully imported hyperliquid packages")
except ImportError as e:
    logger.error(f"Failed to import hyperliquid packages: {e}")
    raise

# Import eth_account
try:
    import eth_account
    logger.info("Successfully imported eth_account")
except ImportError as e:
    logger.error(f"Failed to import eth_account: {e}")
    raise

class GridTrader:
    """
    A grid trading bot that maintains exactly GRID_LEVELS open orders at all times,
    with special handling for filled orders.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the grid trader with configuration from a file.
        
        Args:
            config_path: Path to the configuration file
        """
        # Grid Trading Configuration (can be overridden via environment variables)
        self.GRID_UPPER_BOUNDARY = _env_or_default("GRID_UPPER_BOUNDARY", 1.0, float)
        self.GRID_LOWER_BOUNDARY = _env_or_default("GRID_LOWER_BOUNDARY", 0.99, float)
        self.GRID_LEVELS = _env_or_default("GRID_LEVELS", 40, int)
        self.GRID_ORDER_SIZE = _env_or_default("GRID_ORDER_SIZE", 200, float)
        
        # Trading pair and market configuration
        self.SYMBOL = _env_or_default("GRID_SYMBOL", "FEUSD/USDC", str)
        self.SPOT_ASSET_INDEX = _env_or_default("GRID_SPOT_ASSET_INDEX", 153, int)
        self.MARKET_TYPE = _env_or_default("GRID_MARKET_TYPE", "spot", str).lower()
        self.BASE_ASSET = _env_or_default("GRID_BASE_ASSET", "FEUSD", str)
        self.QUOTE_ASSET = _env_or_default("GRID_QUOTE_ASSET", "USDC", str)
        self.COLLATERAL_ASSET = _env_or_default("GRID_COLLATERAL_ASSET", self.QUOTE_ASSET, str)
        self.PERP_DEX = _env_or_default("GRID_PERP_DEX", "", str)
        self.DEX = self.PERP_DEX if self.MARKET_TYPE == "perp" else ""
        tick_size_value = _get_env_value("GRID_TICK_SIZE")
        self.TICK_SIZE = float(tick_size_value) if tick_size_value not in (None, "") else None
        if self.TICK_SIZE is not None and self.TICK_SIZE <= 0:
            self.TICK_SIZE = None
        logger.info(
            "Grid config: symbol=%s asset_index=%s upper=%s lower=%s levels=%s order_size=%s market=%s base=%s quote=%s collateral=%s perp_dex=%s tick=%s",
            self.SYMBOL,
            self.SPOT_ASSET_INDEX,
            self.GRID_UPPER_BOUNDARY,
            self.GRID_LOWER_BOUNDARY,
            self.GRID_LEVELS,
            self.GRID_ORDER_SIZE,
            self.MARKET_TYPE,
            self.BASE_ASSET,
            self.QUOTE_ASSET,
            self.COLLATERAL_ASSET,
            self.PERP_DEX or "(default)",
            self.TICK_SIZE or "auto",
        )
        
        # Calculate grid step size
        self.GRID_STEP = (self.GRID_UPPER_BOUNDARY - self.GRID_LOWER_BOUNDARY) / self.GRID_LEVELS
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create Ethereum account from private key
        try:
            logger.info("Creating Ethereum account from private key")
            self.account = eth_account.Account.from_key(self.config["secret_key"])
            logger.info("Successfully created Ethereum account")
        except Exception as e:
            logger.error(f"Error creating Ethereum account: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
        self.address = self.config.get("account_address", "") or self.account.address
        logger.info(f"Using address: {self.address}")
        
        # Initialize API clients
        perp_dexs = [self.DEX] if self.DEX else None
        self.info = Info(constants.MAINNET_API_URL, perp_dexs=perp_dexs)
        self.exchange = Exchange(
            self.account,
            constants.MAINNET_API_URL,
            account_address=self.address,
            perp_dexs=perp_dexs
        )

        coin_name = self.info.name_to_coin.get(self.SYMBOL)
        if coin_name:
            self.SPOT_ASSET_INDEX = self.info.coin_to_asset.get(coin_name, self.SPOT_ASSET_INDEX)
        
        # Store active orders
        self.active_orders: Dict[float, Dict[str, Any]] = {}
        
        # Generate grid levels
        self.grid_levels = self.generate_levels()
        if len(self.grid_levels) >= 2:
            computed_step = round(self.grid_levels[1] - self.grid_levels[0], 6)
            if computed_step > 0:
                self.GRID_STEP = computed_step
        
        # Track order history to detect fills
        self.order_history = {}
        
        print(f"Initialized grid trading bot for {self.SYMBOL}")
        print(f"Grid levels: {self.grid_levels}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def generate_levels(self) -> List[float]:
        """
        Generate the grid price levels.
        
        Returns:
            List of price levels sorted from lowest to highest
        """
        # Generate grid levels from lower to upper boundary
        levels = np.linspace(self.GRID_LOWER_BOUNDARY, self.GRID_UPPER_BOUNDARY, self.GRID_LEVELS + 1)
        rounded_levels = [_round_value_to_tick(float(level), self.TICK_SIZE) for level in levels]
        unique_levels = sorted(set(rounded_levels))
        return [round(level, 6) for level in unique_levels]
    
    def get_current_market_price(self) -> float:
        """
        Get the current market price for the trading pair.
        
        Returns:
            Current market price as a float
        """
        print(f"Getting current market price for {self.SYMBOL}...")
        
        try:
            # Try to get the price from token details endpoint
            token_id = f"0x{self.SPOT_ASSET_INDEX:032x}"
            token_details = self._get_token_details(token_id)
            
            if token_details and "midPx" in token_details:
                price = float(token_details["midPx"])
                print(f"Got current price from token details: {price}")
                return price
            
            # Fallback to order book
            order_book = self.info.l2_snapshot(self.SYMBOL)
            
            if order_book and isinstance(order_book, dict) and "levels" in order_book:
                levels = order_book["levels"]
                
                if isinstance(levels, list) and len(levels) >= 2:
                    # In the list format, first element is bids, second is asks
                    bids_data = levels[0]
                    asks_data = levels[1]
                    
                    if bids_data and asks_data:
                        best_bid = float(bids_data[0]["px"])
                        best_ask = float(asks_data[0]["px"])
                        price = (best_bid + best_ask) / 2
                        print(f"Got current price from order book: {price}")
                        return price
        
        except Exception as e:
            print(f"Error getting market price: {e}")
        
        # Default to mid-grid if all methods fail
        price = (self.GRID_UPPER_BOUNDARY + self.GRID_LOWER_BOUNDARY) / 2
        print(f"Using default mid-grid price: {price}")
        return price
    
    def _get_token_details(self, token_id: str) -> Optional[Dict]:
        """Get token details from Hyperliquid API."""
        try:
            import requests
            
            url = "https://api.hyperliquid.xyz/info"
            headers = {"Content-Type": "application/json"}
            payload = {
                "type": "tokenDetails",
                "tokenId": token_id
            }
            
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting token details: {e}")
        
        return None
    
    def _get_perp_account_state(self) -> Optional[Dict]:
        """Fetch perpetual account summary for collateral calculations."""
        if self.MARKET_TYPE != "perp":
            return None
        try:
            return self.info.user_state(self.address, self.DEX)
        except Exception as e:
            print(f"Error getting perp account state: {e}")
            return None
    
    def _get_perp_collateral_balance(self) -> float:
        """Extract available collateral from the perp account state."""
        builder_value = 0.0
        state = self._get_perp_account_state()
        if state:
            withdrawable = state.get("withdrawable")
            if withdrawable is not None:
                try:
                    builder_value = max(builder_value, float(withdrawable))
                except (ValueError, TypeError):
                    pass
            
            margin_summary = state.get("marginSummary") or {}
            account_value = margin_summary.get("accountValue")
            if account_value is not None:
                try:
                    builder_value = max(builder_value, float(account_value))
                except (ValueError, TypeError):
                    pass
        
        spot_value = 0.0
        try:
            spot_user_state = self.info.spot_user_state(self.address)
            for balance in spot_user_state.get("balances", []):
                coin = balance.get("coin")
                if coin == self.COLLATERAL_ASSET:
                    total = float(balance.get("total", 0))
                    hold = float(balance.get("hold", 0))
                    available = max(0.0, total - hold)
                    spot_value = max(spot_value, available)
        except Exception as e:
            print(f"Error getting spot collateral fallback: {e}")
        
        return max(builder_value, spot_value)
    
    def check_balances(self) -> Dict[str, float]:
        """
        Check available balances/collateral for trading.
        
        Returns:
            Dictionary with available balances for configured assets
        """
        balances = {
            self.BASE_ASSET: 0.0,
            self.QUOTE_ASSET: 0.0,
            self.COLLATERAL_ASSET: 0.0,
        }
        
        if self.MARKET_TYPE == "spot":
            try:
                spot_user_state = self.info.spot_user_state(self.address)
                for balance in spot_user_state["balances"]:
                    coin = balance.get("coin", "")
                    total = float(balance.get("total", 0))
                    hold = float(balance.get("hold", 0))
                    available = total - hold
                    
                    if coin == self.BASE_ASSET:
                        balances[self.BASE_ASSET] = available
                    elif coin == self.QUOTE_ASSET:
                        balances[self.QUOTE_ASSET] = available
                
                print(
                    f"Available balances - {self.BASE_ASSET}: {balances[self.BASE_ASSET]}, "
                    f"{self.QUOTE_ASSET}: {balances[self.QUOTE_ASSET]}"
                )
            except Exception as e:
                print(f"Error checking balances: {e}")
        else:
            collateral = self._get_perp_collateral_balance()
            balances[self.COLLATERAL_ASSET] = collateral
            print(f"Available collateral - {self.COLLATERAL_ASSET}: {collateral}")
        
        return balances
    
    def place_initial_orders(self, current_price: float) -> None:
        """
        Place initial grid orders based on the current market price.
        
        Args:
            current_price: Current market price
        """
        print(f"Placing initial orders with current price: {current_price}")
        
        # Get existing orders
        existing_orders = self.get_existing_orders()
        
        # If we already have enough orders, don't place more
        if len(existing_orders) >= self.GRID_LEVELS:
            print(f"Already have {len(existing_orders)} orders. Not placing initial orders.")
            return
        
        # Find the level closest to current price to exclude
        closest_level = self.find_closest_level(current_price)
        print(f"Excluding level closest to price: {closest_level}")
        
        # Place buy orders below current price
        buy_levels = [level for level in self.grid_levels if level < current_price and level != closest_level]
        for level in buy_levels:
            if level not in existing_orders:
                self.place_buy_order(level)
        
        # Place sell orders above current price
        sell_levels = [level for level in self.grid_levels if level > current_price and level != closest_level]
        for level in sell_levels:
            if level not in existing_orders:
                self.place_sell_order(level)
        
        # Ensure we have exactly GRID_LEVELS orders
        self.rebalance_orders(current_price)
    
    def find_closest_level(self, price: float) -> float:
        """
        Find the grid level closest to the given price.
        
        Args:
            price: Price to find closest level to
            
        Returns:
            The grid level closest to the price
        """
        return min(self.grid_levels, key=lambda x: abs(x - price))
    
    def get_existing_orders(self) -> Dict[float, Dict]:
        """
        Get all existing orders for the trading pair.
        
        Returns:
            Dictionary mapping price levels to order details
        """
        existing_orders = {}
        
        try:
            open_orders = self.info.open_orders(self.address, self.DEX)
            
            for order in open_orders:
                if order.get("coin") == self.SYMBOL or order.get("coin") == f"@{self.SPOT_ASSET_INDEX}":
                    price = float(order.get("limitPx", 0))
                    price_key = round(price, 6)
                    order_id = order.get("oid")
                    side = "buy" if order.get("side") == "B" else "sell"
                    size = float(order.get("sz", 0))
                    
                    existing_orders[price_key] = {
                        "id": order_id,
                        "side": side,
                        "price": price_key,
                        "size": size,
                        "status": "active"
                    }
                    
                    # Also update our active_orders tracking
                    self.active_orders[price_key] = existing_orders[price_key]
                    
                    # Add to order history for tracking fills
                    if order_id not in self.order_history:
                        self.order_history[order_id] = {
                            "id": order_id,
                            "price": price_key,
                            "side": side,
                            "size": size,
                            "status": "open",
                            "last_checked": int(time.time())
                        }
            
            print(f"Found {len(existing_orders)} existing orders")
        except Exception as e:
            print(f"Error getting existing orders: {e}")
        
        return existing_orders
    
    def place_buy_order(self, price: float) -> Optional[str]:
        """
        Place a limit buy order at the specified price.
        
        Args:
            price: Price level to place the buy order at
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Round price to 6 decimal places
            price = round(price, 6)
            price = _round_value_to_tick(price, self.TICK_SIZE)
            
            # Check if we already have an order at this price
            if price in self.active_orders:
                print(f"Already have an order at price {price}. Skipping.")
                return None
            
            # Check balances depending on market type
            balances = self.check_balances()
            order_notional = self.GRID_ORDER_SIZE * price
            
            if self.MARKET_TYPE == "spot":
                quote_available = balances.get(self.QUOTE_ASSET, 0.0)
                if quote_available < order_notional:
                    print(
                        f"Insufficient {self.QUOTE_ASSET} balance. "
                        f"Have: {quote_available}, Need: {order_notional}"
                    )
                    return None
            else:
                collateral_available = balances.get(self.COLLATERAL_ASSET, 0.0)
                if collateral_available <= 0:
                    print(
                        f"No detectable {self.COLLATERAL_ASSET} collateral. "
                        "Attempting order; ensure your perp account is funded."
                    )
            
            print(
                f"Placing buy order: {self.GRID_ORDER_SIZE} {self.BASE_ASSET} at "
                f"{price} {self.QUOTE_ASSET}"
            )
            
            # Place the order
            order_result = self.exchange.order(
                self.SYMBOL,
                True,  # isBuy = True for buying
                self.GRID_ORDER_SIZE,
                price,
                {"limit": {"tif": "Gtc"}}  # Good-til-canceled order
            )
            
            if order_result["status"] == "ok":
                status = order_result["response"]["data"]["statuses"][0]
                if "resting" in status:
                    order_id = status["resting"]["oid"]
                    print(f"Buy order placed successfully. Order ID: {order_id}")
                    
                    # Store order information
                    self.active_orders[price] = {
                        "id": order_id,
                        "side": "buy",
                        "price": price,
                        "size": self.GRID_ORDER_SIZE,
                        "status": "active"
                    }
                    
                    # Add to order history for tracking fills
                    self.order_history[order_id] = {
                        "id": order_id,
                        "price": price,
                        "side": "buy",
                        "size": self.GRID_ORDER_SIZE,
                        "status": "open",
                        "last_checked": int(time.time())
                    }
                    
                    return order_id
                elif "filled" in status:
                    order_id = status["filled"]["oid"]
                    print(f"Buy order filled immediately. Order ID: {order_id}")
                    self.on_fill(price, "buy")
                    return order_id
            
            print(f"Error placing buy order: {order_result}")
        except Exception as e:
            print(f"Exception placing buy order: {e}")
        
        return None
    
    def place_sell_order(self, price: float) -> Optional[str]:
        """
        Place a limit sell order at the specified price.
        
        Args:
            price: Price level to place the sell order at
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Round price to 6 decimal places
            price = round(price, 6)
            price = _round_value_to_tick(price, self.TICK_SIZE)
            
            # Check if we already have an order at this price
            if price in self.active_orders:
                print(f"Already have an order at price {price}. Skipping.")
                return None
            
            # Check balances depending on market type
            balances = self.check_balances()
            
            if self.MARKET_TYPE == "spot":
                base_available = balances.get(self.BASE_ASSET, 0.0)
                if base_available < self.GRID_ORDER_SIZE:
                    print(
                        f"Insufficient {self.BASE_ASSET} balance. "
                        f"Have: {base_available}, Need: {self.GRID_ORDER_SIZE}"
                    )
                    return None
            else:
                collateral_available = balances.get(self.COLLATERAL_ASSET, 0.0)
                if collateral_available <= 0:
                    print(
                        f"No detectable {self.COLLATERAL_ASSET} collateral. "
                        "Attempting order; ensure your perp account is funded."
                    )
            
            print(
                f"Placing sell order: {self.GRID_ORDER_SIZE} {self.BASE_ASSET} at "
                f"{price} {self.QUOTE_ASSET}"
            )
            
            # Place the order
            order_result = self.exchange.order(
                self.SYMBOL,
                False,  # isBuy = False for selling
                self.GRID_ORDER_SIZE,
                price,
                {"limit": {"tif": "Gtc"}}  # Good-til-canceled order
            )
            
            if order_result["status"] == "ok":
                status = order_result["response"]["data"]["statuses"][0]
                if "resting" in status:
                    order_id = status["resting"]["oid"]
                    print(f"Sell order placed successfully. Order ID: {order_id}")
                    
                    # Store order information
                    self.active_orders[price] = {
                        "id": order_id,
                        "side": "sell",
                        "price": price,
                        "size": self.GRID_ORDER_SIZE,
                        "status": "active"
                    }
                    
                    # Add to order history for tracking fills
                    self.order_history[order_id] = {
                        "id": order_id,
                        "price": price,
                        "side": "sell",
                        "size": self.GRID_ORDER_SIZE,
                        "status": "open",
                        "last_checked": int(time.time())
                    }
                    
                    return order_id
                elif "filled" in status:
                    order_id = status["filled"]["oid"]
                    print(f"Sell order filled immediately. Order ID: {order_id}")
                    self.on_fill(price, "sell")
                    return order_id
            
            print(f"Error placing sell order: {order_result}")
        except Exception as e:
            print(f"Exception placing sell order: {e}")
        
        return None
    
    def on_fill(self, price: float, side: str) -> None:
        """
        Handle a filled order according to the grid strategy.
        
        Args:
            price: Price level of the filled order
            side: Side of the filled order ('buy' or 'sell')
        """
        print(f"{side.capitalize()} order at {price} was filled.")
        
        # Get current market price
        current_price = self.get_current_market_price()
        
        # Get all active orders to check how many we have
        active_price_levels = set(self.active_orders.keys())
        num_active_orders = len(active_price_levels)
        
        # If we already have GRID_LEVELS orders, don't place a new one
        if num_active_orders >= self.GRID_LEVELS:
            print(f"Already have {num_active_orders} active orders. Not placing a new one after fill.")
            return
        
        if side == "buy":
            # For a filled buy order, place a sell order at the next grid level up
            sell_price = price + self.GRID_STEP
            sell_price = round(sell_price, 6)
            sell_price = _round_value_to_tick(sell_price, self.TICK_SIZE)
            
            # Check if the sell price is within grid boundaries
            if sell_price <= self.GRID_UPPER_BOUNDARY:
                # Check if we already have an order at this price level
                if sell_price not in self.active_orders:
                    print(f"Placing sell order at {sell_price} based on filled buy order at {price}")
                    self.place_sell_order(sell_price)
                else:
                    print(f"Already have an active order at price {sell_price}. Skipping.")
            else:
                print(f"Sell price {sell_price} is outside grid boundaries. Skipping.")
        
        elif side == "sell":
            # For a filled sell order, place a buy order at the next grid level down
            buy_price = price - self.GRID_STEP
            buy_price = round(buy_price, 6)
            buy_price = _round_value_to_tick(buy_price, self.TICK_SIZE)
            
            # Check if the buy price is within grid boundaries
            if buy_price >= self.GRID_LOWER_BOUNDARY:
                # Check if we already have an order at this price level
                if buy_price not in self.active_orders:
                    print(f"Placing buy order at {buy_price} based on filled sell order at {price}")
                    self.place_buy_order(buy_price)
                else:
                    print(f"Already have an active order at price {buy_price}. Skipping.")
            else:
                print(f"Buy price {buy_price} is below lower boundary {self.GRID_LOWER_BOUNDARY}. Not placing buy order.")
    
    def scan_for_fills(self) -> None:
        """Scan for filled orders and handle them according to the grid strategy."""
        print("Scanning for filled orders...")
        
        try:
            # Get all open orders
            open_orders = self.info.open_orders(self.address, self.DEX)
            current_order_ids = set()
            
            # Track which price levels currently have active orders
            active_price_levels = set()
            
            # Update our active orders tracking
            new_active_orders = {}
            
            for order in open_orders:
                if order.get("coin") == self.SYMBOL or order.get("coin") == f"@{self.SPOT_ASSET_INDEX}":
                    order_id = order.get("oid")
                    price = float(order.get("limitPx", 0))
                    price_key = round(price, 6)
                    side = "buy" if order.get("side") == "B" else "sell"
                    size = float(order.get("sz", 0))
                    
                    current_order_ids.add(order_id)
                    active_price_levels.add(price_key)
                    
                    new_active_orders[price_key] = {
                        "id": order_id,
                        "side": side,
                        "price": price_key,
                        "size": size,
                        "status": "active"
                    }
            
            # Replace our active_orders with this accurate snapshot
            self.active_orders = new_active_orders
            
            # Check for orders that were in our history but are no longer open (filled orders)
            filled_orders = []
            current_time = int(time.time())
            five_minutes_ago = current_time - 300  # Check orders from the last 5 minutes
            
            for order_id, order_info in self.order_history.items():
                # Only consider orders we've checked recently
                if order_info["last_checked"] >= five_minutes_ago and order_info["status"] == "open":
                    if order_id not in current_order_ids:
                        # This order was open before but is no longer open, so it must have been filled
                        order_info["status"] = "filled"
                        filled_orders.append(order_info)
                        print(f"Detected filled order: {order_info}")
            
            # Update last check time for all orders in history
            for order_id in self.order_history:
                self.order_history[order_id]["last_checked"] = current_time
            
            # Handle filled orders
            for order in filled_orders:
                price = order["price"]
                side = order["side"]
                self.on_fill(price, side)
            
            # Rebalance orders if needed
            current_price = self.get_current_market_price()
            self.rebalance_orders(current_price)
            
        except Exception as e:
            print(f"Error scanning for fills: {e}")
    
    def rebalance_orders(self, current_price: float) -> None:
        """
        Rebalance orders to ensure we have exactly GRID_LEVELS orders.
        
        Args:
            current_price: Current market price
        """
        # Get all active orders
        active_price_levels = set(self.active_orders.keys())
        num_active_orders = len(active_price_levels)
        
        print(f"Current active price levels: {sorted(list(active_price_levels))}")
        print(f"Current number of active orders: {num_active_orders} out of {self.GRID_LEVELS} desired")
        
        # If we already have the desired number of orders, don't add more
        if num_active_orders >= self.GRID_LEVELS:
            print(f"Already have {num_active_orders} active orders. Not adding more.")
            return
        
        # Calculate how many orders we need to add
        orders_to_add = self.GRID_LEVELS - num_active_orders
        print(f"Need to add {orders_to_add} orders to reach {self.GRID_LEVELS} total")
        
        # Find the level closest to current price to exclude
        closest_level = self.find_closest_level(current_price)
        
        # Sort grid prices
        sorted_grid_prices = sorted(self.grid_levels)
        
        # Filter out prices that already have orders or are the closest to current price
        available_prices = [price for price in sorted_grid_prices 
                           if round(price, 6) not in active_price_levels 
                           and price != closest_level]
        
        # If we have no available prices, we're done
        if not available_prices:
            print("No available price levels to place orders. Skipping.")
            return
        
        # Split available prices into buy and sell based on current price
        buy_prices = [price for price in available_prices if price < current_price]
        sell_prices = [price for price in available_prices if price > current_price]
        
        # Sort buy prices in descending order (closest to current price first)
        buy_prices.sort(reverse=True)
        # Sort sell prices in ascending order (closest to current price first)
        sell_prices.sort()
        
        # Determine how many buy and sell orders to place
        buy_orders_to_add = min(len(buy_prices), orders_to_add // 2 + (orders_to_add % 2))
        sell_orders_to_add = min(len(sell_prices), orders_to_add - buy_orders_to_add)
        
        # If we couldn't place all sell orders, add more buy orders if possible
        if sell_orders_to_add < orders_to_add - buy_orders_to_add:
            buy_orders_to_add = min(len(buy_prices), orders_to_add - sell_orders_to_add)
        
        # Place buy orders
        for i in range(buy_orders_to_add):
            if i < len(buy_prices):
                price = buy_prices[i]
                print(f"Placing buy order at grid level {price}")
                self.place_buy_order(price)
        
        # Place sell orders
        for i in range(sell_orders_to_add):
            if i < len(sell_prices):
                price = sell_prices[i]
                print(f"Placing sell order at grid level {price}")
                self.place_sell_order(price)
    
    def cancel_all_orders(self) -> None:
        """Cancel all existing orders for the trading pair"""
        print("Canceling all existing orders...")
        try:
            # Get all open orders
            open_orders = self.info.open_orders(self.address, self.DEX)
            print("Open orders:")
            print(json.dumps(open_orders, indent=2))
            
            # Track how many orders were canceled
            canceled_count = 0
            
            # Filter orders for our specific trading pair
            symbol_orders = []
            for order in open_orders:
                # For spot trading, the coin format might be '@153' for FEUSD
                if isinstance(order, dict) and "coin" in order:
                    if order["coin"] == self.SYMBOL or order["coin"] == f"@{self.SPOT_ASSET_INDEX}":
                        symbol_orders.append(order)
            
            print(f"Found {len(symbol_orders)} {self.SYMBOL} orders to cancel")
            
            # Prepare batch cancel request
            cancels = []
            for order in symbol_orders:
                order_id = order.get("oid")
                if order_id:
                    # For spot trading, we need to use the asset ID
                    cancels.append({"a": self.SPOT_ASSET_INDEX, "o": order_id})
            
            # If we have orders to cancel, send a batch cancel request
            if cancels:
                print(f"Canceling {len(cancels)} orders in batch")
                try:
                    # Use the exchange API to cancel orders in batch
                    cancel_action = {"type": "cancel", "cancels": cancels}
                    cancel_result = self.exchange.action(cancel_action)
                    print(f"Batch cancel result: {cancel_result}")
                    
                    # Check if the cancellation was successful
                    if cancel_result["status"] == "ok":
                        statuses = cancel_result["response"]["data"]["statuses"]
                        for i, status in enumerate(statuses):
                            if status == "success":
                                canceled_count += 1
                            else:
                                print(f"Failed to cancel order {cancels[i]['o']}: {status}")
                except Exception as e:
                    print(f"Error in batch cancel: {e}")
                    print(f"Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    
                    # Fallback to individual cancellations if batch fails
                    print("Falling back to individual cancellations...")
                    for order in symbol_orders:
                        try:
                            order_id = order.get("oid")
                            if order_id:
                                print(f"Canceling order ID: {order_id}")
                                # Try using the direct cancel method
                                cancel_result = self.exchange.cancel(self.SYMBOL, order_id)
                                print(f"Cancel result: {cancel_result}")
                                if cancel_result["status"] == "ok":
                                    canceled_count += 1
                        except Exception as e:
                            print(f"Error canceling order: {e}")
            
            print(f"Canceled {canceled_count} orders")
            
            # Wait a moment for cancellations to process
            time.sleep(2)
            
            # Clear our tracking
            self.active_orders = {}
            
        except Exception as e:
            print(f"Error in cancel_all_orders: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
    
    def run(self) -> None:
        """Run the grid trading bot."""
        print("Starting grid trading bot...")
        
        # Cancel all existing orders to start fresh
        self.cancel_all_orders()
        
        # Get current market price
        current_price = self.get_current_market_price()
        
        # Place initial orders
        self.place_initial_orders(current_price)
        
        # Main loop
        while True:
            try:
                # Scan for filled orders
                self.scan_for_fills()
                
                # Sleep to avoid API rate limits
                print("Sleeping for 15 seconds...")
                time.sleep(15)
            except KeyboardInterrupt:
                print("Bot stopped by user")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(15)

if __name__ == "__main__":
    # Create and run the grid trader
    trader = GridTrader("config.json")
    trader.run()
