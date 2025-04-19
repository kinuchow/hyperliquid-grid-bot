# Hyperliquid Grid Trading Bot

This bot implements a grid trading strategy for the FEUSD/USDC trading pair on Hyperliquid DEX. It automatically places buy and sell orders according to a predefined grid, creating a network of orders that can profit from price oscillations within a range.

## Grid Trading Strategy

The bot follows these rules:
1. After each sell order is filled, a new buy order is automatically placed one grid level below the executed sell price.
2. After each buy order is filled, a new sell order is automatically placed one grid level above the executed buy price.
3. The bot maintains exactly 40 active orders at all times (configurable via GRID_LEVELS).
4. The level closest to the current market price is excluded when placing initial orders.

## Configuration

All configuration parameters can be easily modified in the `grid_trader.py` file:

```python
# Grid Trading Configuration
self.GRID_UPPER_BOUNDARY = 1.0    # Upper price boundary
self.GRID_LOWER_BOUNDARY = 0.99   # Lower price boundary
self.GRID_LEVELS = 40             # Number of active orders to maintain
self.GRID_ORDER_SIZE = 200        # Size of each order in base currency

# Trading pair configuration
self.SYMBOL = "FEUSD/USDC"        # Trading pair symbol
self.SPOT_ASSET_INDEX = 153       # Spot asset index on Hyperliquid
```

### Spot Asset Configuration

The bot is currently configured for FEUSD/USDC (spot asset index 153), but can be adapted to trade any spot pair on Hyperliquid by changing the `SYMBOL` and `SPOT_ASSET_INDEX` values. You can find the asset index for other trading pairs in the Hyperliquid documentation or API.

This creates a grid with 40 levels between 0.99 and 1.0, with each grid step being approximately 0.00025. The increased number of grid levels and tighter grid spacing allows for more frequent trading opportunities.

## Requirements

- Python 3.10+
- hyperliquid-python-sdk
- pandas
- numpy
- matplotlib
- requests
- eth_account

## Deployment on Railway

Railway is a modern platform that makes it easy to deploy and run your grid trading bot 24/7 without having to manage servers. Here's how to deploy your bot on Railway:

### Prerequisites

1. Create a [Railway account](https://railway.app/)
2. Install the [Railway CLI](https://docs.railway.app/develop/cli)
3. Fork this repository on GitHub

### Step 1: Prepare your project

Create the following files in your project directory if they don't already exist:

#### requirements.txt

```
hyperliquid-python-sdk>=0.0.8
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
requests>=2.28.0
eth-account>=0.8.0
```

#### Procfile

```
worker: python main.py
```

#### main.py

```python
#!/usr/bin/env python3
"""
Main entry point for the Hyperliquid Grid Trading Bot
"""

import os
import time
import logging
from grid_trader import GridTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Get config path from environment variable or use default
    config_path = os.environ.get("CONFIG_PATH", "config.json")
    
    # Create grid trader instance
    try:
        trader = GridTrader(config_path)
        logger.info("Grid trader initialized successfully")
        
        # Initialize grid
        trader.initialize_grid()
        logger.info("Grid initialized successfully")
        
        # Main loop
        while True:
            try:
                # Update grid
                trader.update_grid()
                logger.info("Grid updated successfully")
                
                # Sleep for a while
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error updating grid: {e}")
                time.sleep(60)  # Wait before retrying
    except Exception as e:
        logger.error(f"Error initializing grid trader: {e}")

if __name__ == "__main__":
    main()
```

### Step 2: Set up Railway

1. Login to Railway using the CLI:
   ```
   railway login
   ```

2. Initialize your project:
   ```
   railway init
   ```

3. Create a new Railway project:
   ```
   railway project create
   ```

4. Link your local project to the Railway project:
   ```
   railway link
   ```

### Step 3: Configure Environment Variables

Instead of using a local config.json file, you'll need to set your private key as an environment variable in Railway:

1. Go to your project on the Railway dashboard
2. Navigate to the "Variables" tab
3. Add the following variables:
   - `SECRET_KEY`: Your Hyperliquid private key
   - `ACCOUNT_ADDRESS`: (Optional) Your Hyperliquid address

### Step 4: Deploy to Railway

1. Deploy your project:
   ```
   railway up
   ```

2. Monitor your deployment on the Railway dashboard

### Step 5: Verify Deployment

1. Check the logs in the Railway dashboard to ensure your bot is running correctly
2. Your grid trading bot should now be running 24/7 on Railway

### Important Notes

- Railway's free tier has limitations, so you may need to upgrade to a paid plan for continuous operation
- Always monitor your bot's performance and be prepared to stop it if necessary
- Never share your private key or config file with anyone
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
