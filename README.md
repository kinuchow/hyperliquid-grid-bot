# Hyperliquid Grid Trading Bot

This bot implements a grid trading strategy for the FEUSD/USDC trading pair on Hyperliquid DEX. It automatically places buy and sell orders according to a predefined grid, creating a network of orders that can profit from price oscillations within a range.

## Grid Trading Strategy

The bot follows these rules:
1. After each sell order is filled, a new buy order is automatically placed one grid level below the executed sell price.
2. After each buy order is filled, a new sell order is automatically placed one grid level above the executed buy price.
3. The bot maintains exactly 5 active orders at all times (configurable via GRID_LEVELS).
4. The level closest to the current market price is excluded when placing initial orders.

## Configuration

All configuration parameters can be easily modified in the `grid_trader.py` file:

```python
# Grid Trading Configuration
self.GRID_UPPER_BOUNDARY = 1.001  # Upper price boundary
self.GRID_LOWER_BOUNDARY = 0.999  # Lower price boundary
self.GRID_LEVELS = 5              # Number of active orders to maintain
self.GRID_ORDER_SIZE = 20         # Size of each order in base currency

# Trading pair configuration
self.SYMBOL = "FEUSD/USDC"        # Trading pair symbol
self.SPOT_ASSET_INDEX = 153       # Spot asset index on Hyperliquid
```

### Spot Asset Configuration

The bot is currently configured for FEUSD/USDC (spot asset index 153), but can be adapted to trade any spot pair on Hyperliquid by changing the `SYMBOL` and `SPOT_ASSET_INDEX` values. You can find the asset index for other trading pairs in the Hyperliquid documentation or API.

This creates a grid with 5 levels between 0.999 and 1.001, with each grid step being approximately 0.0004.

## Requirements

- Python 3.10+
- hyperliquid-python-sdk
- numpy

## Installation

1. Install the required packages:
```
pip install hyperliquid-python-sdk numpy
```

2. Edit the `config.json` file with your Hyperliquid private key:
```json
{
  "secret_key": "YOUR_PRIVATE_KEY_HERE",
  "account_address": ""  // Optional, leave empty to use the address derived from the private key
}
```

## Usage

1. Make sure you have sufficient FEUSD and USDC in your Hyperliquid spot account.
2. Run the bot:
```
python feusd_grid_bot.py
```

## How It Works

1. The bot initializes by placing buy orders below the current market price and sell orders above it.
2. It continuously monitors the status of all active orders.
3. When a buy order is filled, it places a new sell order one grid level above.
4. When a sell order is filled, it places a new buy order one grid level below.
5. This process continues indefinitely, allowing you to profit from price movements within the grid range.

## Risk Management

- The bot only operates within the defined price range (0.92 to 0.96).
- It checks your available balances before placing orders.
- All orders are limit orders to ensure you get the desired price.

## Important Notes

- This bot requires you to have sufficient FEUSD and USDC in your Hyperliquid spot account.
- The bot will continue running until manually stopped (Ctrl+C).
- Grid trading works best in sideways (range-bound) markets.
- You may need to adjust the grid parameters based on market conditions.
- Always monitor the bot's operation, especially during volatile market conditions.

## Customization

You can modify the grid parameters in the `feusd_grid_bot.py` file:
```python
# Grid Trading Configuration
GRID_UPPER_BOUNDARY = 0.96
GRID_LOWER_BOUNDARY = 0.92
GRID_LEVELS = 50
GRID_ORDER_SIZE = 100
```

## Disclaimer

This bot is provided for educational purposes only. Use at your own risk. Trading cryptocurrency carries significant risk, and you should never trade with funds you cannot afford to lose.
