#!/usr/bin/env python3
"""
Wrapper script to ensure all dependencies are properly imported
before running the Hyperliquid Grid Trading Bot
"""

import os
import sys
import logging
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_imports():
    """Check if all required packages are imported correctly"""
    required_packages = [
        "eth_account",
        "hyperliquid.exchange",
        "hyperliquid.info",
        "hyperliquid.utils.constants"
    ]
    
    all_imported = True
    
    for package in required_packages:
        try:
            logger.info(f"Checking import for {package}")
            if "." in package:
                parts = package.split(".")
                module = importlib.import_module(parts[0])
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                importlib.import_module(package)
            logger.info(f"Successfully imported {package}")
        except ImportError as e:
            logger.error(f"Failed to import {package}: {e}")
            all_imported = False
    
    return all_imported

def main():
    """Main function"""
    logger.info("Starting Hyperliquid Grid Trading Bot wrapper")
    
    # Check imports
    if not check_imports():
        logger.error("Failed to import required packages")
        logger.error("Please ensure all packages in requirements.txt are installed")
        sys.exit(1)
    
    # Import and run the main module
    logger.info("All imports successful, running main module")
    import main as grid_bot_main
    grid_bot_main.main()

if __name__ == "__main__":
    main()
