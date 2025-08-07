#!/usr/bin/env python3
"""
Startup script for Fake Review Detection API.

This script provides a convenient way to start the API server with
proper configuration and environment setup.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from src import utils

def setup_environment():
    """Set up environment variables and configuration."""
    # Set up logging
    logger = utils.setup_logging(__name__)
    
    # Ensure required directories exist
    required_dirs = [
        'logs',
        'artifacts/models',
        'data/raw',
        'data/processed'
    ]
    
    for dir_path in required_dirs:
        utils.ensure_dir(PROJECT_ROOT / dir_path)
    
    # Set environment variables for multiprocessing metrics
    os.environ.setdefault('prometheus_multiproc_dir', str(PROJECT_ROOT / 'logs' / 'metrics'))
    
    return logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start Fake Review Detection API')
    
    parser.add_argument(
        '--host', 
        default=None,
        help='Host to bind to (default: from settings)'
    )
    
    parser.add_argument(
        '--port', 
        type=int,
        default=None,
        help='Port to bind to (default: from settings)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: from settings)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload on code changes'
    )
    
    parser.add_argument(
        '--model-path',
        help='Path to model file'
    )
    
    parser.add_argument(
        '--preprocessor-path',
        help='Path to preprocessor file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=None,
        help='Logging level'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup environment
    logger = setup_environment()
    
    # Get configuration
    host = args.host or utils.get_setting('api.host', '0.0.0.0')
    port = args.port or utils.get_setting('api.port', 8000)
    workers = args.workers or utils.get_setting('api.workers', 1)
    debug = args.debug or utils.get_setting('api.debug', False)
    log_level = args.log_level or utils.get_setting('logging.level', 'INFO')
    
    # Override settings if command line arguments are provided
    if args.model_path:
        os.environ['FAKE_REVIEW_MODEL_PATH'] = args.model_path
    if args.preprocessor_path:
        os.environ['FAKE_REVIEW_PREPROCESSOR_PATH'] = args.preprocessor_path
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    logger.info("="*60)
    logger.info("Starting Fake Review Detection API")
    logger.info("="*60)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Debug: {debug}")
    logger.info(f"Log Level: {log_level}")
    
    if args.model_path:
        logger.info(f"Model Path: {args.model_path}")
    if args.preprocessor_path:
        logger.info(f"Preprocessor Path: {args.preprocessor_path}")
    
    logger.info("="*60)
    
    try:
        # Start the server
        uvicorn.run(
            "api.app:app",
            host=host,
            port=port,
            workers=workers if not (debug or args.reload) else 1,
            log_level=log_level.lower(),
            debug=debug,
            reload=args.reload or debug,
            access_log=True,
            reload_dirs=[str(PROJECT_ROOT)] if args.reload or debug else None,
            reload_includes=["*.py"] if args.reload or debug else None
        )
    except KeyboardInterrupt:
        logger.info("Shutting down API server...")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
