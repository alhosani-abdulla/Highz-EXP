#!/usr/bin/env python3
"""
Utility functions for the Digital Spectrometer project, 
including logging setup and other common I/O operations.
"""
from datetime import datetime
import logging

def setup_logging(level=logging.INFO, output_file=False):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if output_file:
        file_name = f'file_compressor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(filename=file_name)