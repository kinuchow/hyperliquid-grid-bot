#!/usr/bin/env python3
"""
Main entry point for the Hyperliquid Grid Trading Bot
"""

import os
import time
import json
import logging
from typing import Dict
import eth_account
from grid_trader import GridTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _load_env_file(path: str) -> Dict[str, str]:
    """Load simple KEY=VALUE pairs from a .env-style file."""
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

def _lookup_env_value(keys, env_file_values):
    """Resolve the first available value for a list of possible env keys."""
    for key in keys:
        if key in os.environ and os.environ.get(key):
            return os.environ.get(key)
    for key in keys:
        if env_file_values.get(key):
            logger.info("Loaded %s from env file", key)
            return env_file_values[key]
    return None

def create_config_from_env():
    """Create a config.json file from environment variables"""
    env_file_path = os.environ.get("GRID_ENV_FILE", ".env")
    env_file_values = _load_env_file(env_file_path)
    
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
    possible_vault_keys = [
        "vault_address", "VAULT_ADDRESS",
        "subaccount_address", "SUBACCOUNT_ADDRESS",
        "RAILWAY_VAULT_ADDRESS", "RAILWAY_VAR_VAULT_ADDRESS",
        "RAILWAY_SUBACCOUNT_ADDRESS", "RAILWAY_VAR_SUBACCOUNT_ADDRESS"
    ]
    
    # Find secret_key
    secret_key = _lookup_env_value(possible_secret_keys, env_file_values)
    if secret_key:
        logger.info("Found secret key in environment data")
    
    # Find account_address
    account_address = _lookup_env_value(possible_address_keys, env_file_values) or ""
    if account_address:
        logger.info("Found account address in environment data")
    
    # Find optional vault/subaccount address
    vault_address = _lookup_env_value(possible_vault_keys, env_file_values) or ""
    if vault_address:
        logger.info("Found vault/subaccount address in environment data")
    
    if not secret_key:
        logger.error("No secret key found in environment variables")
        raise ValueError("No secret key found in environment variables")
    
    config = {
        "secret_key": secret_key,
        "account_address": account_address,
        "vault_address": vault_address
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
            # Get config path from environment variable or use default,
            # and fall back to env vars if the file is missing.
            config_path = os.environ.get("CONFIG_PATH", "config.json")
            if not os.path.exists(config_path):
                logger.warning("Config file %s not found, building it from environment variables", config_path)
                config_path = create_config_from_env()
            
        # Create grid trader instance
        trader = GridTrader(config_path)
        logger.info("Grid trader initialized successfully")
        
        # Run the bot loop
        trader.run()
    except Exception as e:
        logger.error(f"Error initializing grid trader: {e}")

if __name__ == "__main__":
    main()
