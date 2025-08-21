
"""
utils/logging_config.py - Centralized logging configuration for the application.
"""
import logging.config
import sys

def setup_logging(log_level: str = "INFO"):
    """Sets up the logging configuration for the entire application."""
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': sys.stdout,
            },
        },
        'root': {
            'handlers': ['console'],
            'level': log_level,
        },
        'loggers': {
            'httpx': {
                'handlers': ['console'],
                'level': 'WARNING', # Reduce verbosity of HTTP libraries
                'propagate': False
            },
            'httpcore': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False
            },
             'huggingface_hub': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False
            },
        }
    }
    logging.config.dictConfig(LOGGING_CONFIG)
    logging.info(f"Logging configured with level {log_level}")
