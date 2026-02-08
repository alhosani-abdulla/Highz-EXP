# Highz-EXP
This is a preliminary version for plotting and analyzing data collected from the HighZ-EXP experiment. It includes modules for loading files, processing reflection data, plotting spectra, and converting units.

## Table of Contents
- [Related Repositories](#related-repositories)
- [Installation](#installation)
- [Usage](#usage)
- [Modules Overview](#modules-overview)

## Related Repositories

The HighZ-EXP project has been split into focused component repositories:

- **[adf4351-controller](https://github.com/alhosani-abdulla/adf4351-controller)** - Arduino controller programs for the ADF4351 PLL frequency synthesizer used as the Local Oscillator
- **[highz-filterbank](https://github.com/alhosani-abdulla/highz-filterbank)** - Multi-channel filterbank spectrometer for detecting the 21-cm cosmological signal from Cosmic Dawn


## Installation

### From GitHub (for Google Colab or local installation)

```python
!pip install git+https://github.com/alhosani-abdulla/Highz-EXP.git
```
Noe that `-e` flat does not work in Google Colab, as it's designed for local dev where changes to the source code is immediately reflected in the installed package without reinstallation. Google Colab does not provide persistent access to the source code.

### For development (editable install)

```bash
git clone https://github.com/alhosani-abdulla/Highz-EXP.git
cd Highz-EXP
pip install -e .
```

## Usage

```python
# Import the package (note: use underscore, not hyphen)
import highz_exp
from highz_exp import file_load, unit_convert
```

If run into import issues, try 
```bash
pip install --force-reinstall git+https://github.com/alhosani-abdulla/Highz-EXP.git
```

### Best Practices
To get familiar with what this package can do and the way it's most commonly used, we recommend going through the example notebooks in the [notebooks](notebooks/) directory.

Scripts used in our data collection/visualization are available in the [scripts](scripts/) directory.

## Modules Overview

### Key Classes:
- `Spectrum`: Class for handling and plotting spectral data. It also contains basic processing utilities (e.g., averaging, smoothing).
```python
from highz_exp.spec_class import Spectrum
spectrum = Spectrum(frequency_axis, spectral_data, name='example_spectrum', metadata={'obs_time': '2024-01-01T00:00:00Z'})
spectrum.smooth(window=5) # To smooth the data
spectrum.plot() # To visualize
```

### Helper modules:
- `spec_proc`: Functions for processing spectral data (e.g., averaging, smoothing, downsampling).
- `plotter`: Functions for plotting spectra and related visualizations. Commonly used functions are `plot_spectra` and `plot_waterfall`. 