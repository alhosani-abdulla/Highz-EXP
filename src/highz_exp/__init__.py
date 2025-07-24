"""
Highz-EXP: A Python package for analyzing data from the HighZ-EXP experiment.

This package includes modules for:
- Loading files
- Processing reflection data  
- Plotting spectra
- Converting units
"""

__version__ = "0.1.0"

# Import main modules for easier access
from . import file_load
from . import spec_plot  
from . import unit_convert

# Import reflection_proc conditionally (requires scikit-rf)
try:
    from . import reflection_proc
except ImportError:
    # scikit-rf not available
    pass

__all__ = ['file_load', 'spec_plot', 'unit_convert']

# Add reflection_proc to __all__ if it was successfully imported
if 'reflection_proc' in locals():
    __all__.append('reflection_proc')