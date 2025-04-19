#!/usr/bin/env python3
"""
Main entry point for the Hyperliquid Grid Trading Bot
"""

import os
import time
import json
import logging
import eth_account
from grid_trader import GridTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_config_from_env():
    """Create a config.json file from environment variables"""
    # Try different possible environment variable names for secret_key
    possible_secret_keys = [
        "secret_key", "SECRET_KEY",
        "RAILWAY_SECRET_KEY", "RAILWAY_VAR_SECRET_KEY",
        "HYPERLIQUID_SECRET_KEY", "PRIVATE_KEY"
    ]
    
    # Try different possible environment variable names for account_address
    possible_address_keys = [
        "account_address", "ACCOUNT_ADDRESS",
        "RAILWAY_ACCOUNT_ADDRESS", "RAILWAY_VAR_ACCOUNT_ADDRESS",
        "HYPERLIQUID_ADDRESS", "ADDRESS"
    ]
    
    # Find secret_key
    secret_key = None
    for key in possible_secret_keys:
        if key in os.environ:
            secret_key = os.environ.get(key)
            logger.info(f"Found secret key using variable: {key}")
            break
    
    # Find account_address
    account_address = ""
    for key in possible_address_keys:
        if key in os.environ:
            account_address = os.environ.get(key)
            logger.info(f"Found account address using variable: {key}")
            break
    
    if not secret_key:
        logger.error("No secret key found in environment variables")
        raise ValueError("No secret key found in environment variables")
    
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
