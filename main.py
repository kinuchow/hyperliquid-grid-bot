#!/usr/bin/env python3
"""
Main entry point for the Hyperliquid Grid Trading Bot
"""

import os
import time
import json
import logging
from grid_trader import GridTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_config_from_env():
    """Create a config.json file from environment variables"""
    secret_key = os.environ.get("SECRET_KEY")
    account_address = os.environ.get("ACCOUNT_ADDRESS", "")
    
    if not secret_key:
        raise ValueError("SECRET_KEY environment variable is required")
    
    config = {
        "secret_key": secret_key,
        "account_address": account_address
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f)
    
    logger.info("Created config.json from environment variables")
    return "config.json"

def main():
    try:
        # Check if we're running on Railway
        if "RAILWAY_ENVIRONMENT" in os.environ:
            logger.info("Running on Railway, creating config from environment variables")
            config_path = create_config_from_env()
        else:
            # Get config path from environment variable or use default
            config_path = os.environ.get("CONFIG_PATH", "config.json")
            
        # Create grid trader instance
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
