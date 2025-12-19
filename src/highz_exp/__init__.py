"""
Highz-EXP: A Python package for analyzing data from the HighZ-EXP experiment.

This package includes modules for:
- Loading files
- Processing reflection data  
- Plotting spectra
- Converting units
- Filterbank spectrometer data analysis
"""

__version__ = "0.1.0"

# Import main modules for easier access
from . import file_load
from . import plotter
from . import unit_convert
from . import filterbank
from . import filter_plotting

# Import reflection_proc conditionally (requires scikit-rf)
try:
    from . import reflection_proc
    from . import spec_proc
except ImportError:
    # scikit-rf not available
    pass

__all__ = ['file_load', 'plotter', 'unit_convert', 'filterbank', 'filter_plotting']

# Add reflection_proc to __all__ if it was successfully imported
if 'reflection_proc' in locals():
    __all__.append('reflection_proc')
if 'spec_proc' in locals():
    __all__.append('spec_proc')