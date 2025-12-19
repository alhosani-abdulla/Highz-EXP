"""
High-Z Filterbank Spectrometer Data Processing

This package provides tools for consolidating, visualizing, and analyzing data
from the High-Z filterbank spectrometer (NOT the digital spectrometer).

Main modules:
- consolidation: Data consolidation pipeline
- visualization: Plotting and dashboard tools
- analysis: Statistical analysis utilities
"""

__version__ = "1.0.0"
__author__ = "High-Z Collaboration"

from . import consolidation
from . import visualization
from . import analysis

__all__ = ['consolidation', 'visualization', 'analysis']
