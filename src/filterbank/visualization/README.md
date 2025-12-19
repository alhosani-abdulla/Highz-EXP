# Filterbank Data Visualization

Tools for creating publication-quality plots and interactive visualizations of consolidated filterbank spectrometer data.

## Overview

The visualization module provides:
- **Waterfall Plots**: Spectrogram-style time-frequency plots showing power evolution
- **Historical Viewer**: Web-based interactive dashboard for real-time data exploration
- **Single Spectrum**: Quick plotting utility for individual spectra
- **Interactive Features**: Zoom, pan, frequency range selection per-filter analysis

## Usage

### Waterfall Plots

Create spectrogram/waterfall plots from consolidated data:

```python
from filterbank.visualization import waterfall

# Plot state 1 for November 6
day_dir = Path("/path/to/consolidated/20251106")
output_dir = Path("./plots")

waterfall.plot_day_waterfall(
    day_dir,
    states=['1'],
    output_dir=output_dir,
    freq_range=(0, 250),           # MHz range
    filter_num=None,                # None for all filters, or specific filter 0-20
    interactive=True,
    verbose=True
)
```

#### Command Line Usage

```bash
# Plot state 1
python -m filterbank.visualization.waterfall /path/to/20251106 --state 1

# Plot multiple states
python -m filterbank.visualization.waterfall /path/to/20251106 --state 1 2 3

# Plot single filter with frequency range
python -m filterbank.visualization.waterfall /path/to/20251106 --state 1 --filter 10 --freq-range 50 200

# Plot all states to output directory
python -m filterbank.visualization.waterfall /path/to/20251106 --all-states --output ./plots
```

#### Options

- `--state N [N ...]`: Specify states to plot (e.g., `1 2 3`)
- `--all-states`: Plot all available states
- `--filter N`: Limit to single filter (0-20)
- `--freq-range MIN MAX`: Frequency range in MHz
- `--output DIR`: Save plots to directory
- `--power-range VMIN VMAX`: Set colorbar limits (dBm)
- `--no-interactive`: Disable zoom/pan (batch mode)
- `--verbose`: Print detailed information

### Historical Viewer

Interactive web dashboard for exploring data with multiple simultaneous plots:

```bash
# Start the dashboard
python -m filterbank.visualization.historical /path/to/consolidated/20251106
```

Features:
- File/state/cycle selector
- Per-filter calibration application (if available)
- Four simultaneous views of data
- S21 correction loading from highz-filterbank repo
- Real-time updates and zoom/pan

### Single Spectrum Plotting

Quick utility for plotting individual spectra:

```python
from filterbank.visualization import single_spectrum

# Plot a single spectrum
spectrum_file = Path("/path/to/state_1.fits")
output_path = Path("spectrum.png")

single_spectrum.plot_spectrum(
    spectrum_file,
    spectrum_index=0,
    output_path=output_path,
    show=True
)
```

## Calibration & Corrections

### Filter Calibration

Per-filter calibration converts ADC counts to power using:
- High power (0dBm) and low power (-9dBm) reference measurements
- Per-filter slope and intercept
- Voltage-to-power conversion

### S21 Corrections

Optional S21 loss corrections from filter network characterization:
- Loaded from `highz-filterbank/characterization/s_parameters/`
- Applied as frequency-dependent loss correction
- Improves power measurement accuracy

### Filter Normalization

Automatic normalization brings all 21 filters to comparable power levels:
- Uses reference frequency region (50-80 MHz)
- Per-filter offset calculated from mean power
- Improves visual consistency across filter bank

## Data Quality

The visualization module includes quality checks:
- **Position-based filtering**: Excludes boundary spectra (potential sync issues)
- **Spectrum quality assessment**: Detects shifted or corrupted spectra
- **Reference comparison**: Compares to known-good reference spectra

## File Format

### FITS Input Files
State files (`state_N.fits`) contain:
- **Binary Table** with columns:
  - `SPECTRUM_TIMESTAMP`: Time of spectrum collection
  - `DATA_CUBE`: Reshaped array of ADC counts (21 filters × 144 LO points)
  - `LO_FREQUENCIES` (optional): Actual LO frequencies (handles circular buffer bug)

- **Header** keywords:
  - `STATE`: State number
  - `N_FILTERS`: Number of filters (typically 21)
  - `N_LO_PTS`: Number of LO frequency points (typically 144)
  - `CYCLE_ID`: Cycle identifier
  - `ANTENNA`: Antenna name

## Module Reference

### `waterfall.py`
- `WaterfallData`: Container for waterfall plot data
- `load_state_file()`: Load and process consolidated state file
- `create_waterfall_grid()`: Interpolate data onto regular frequency grid
- `plot_waterfall()`: Create single waterfall plot
- `plot_day_waterfall()`: Batch processing across cycles and states

### `historical.py`
- `SpectrumViewer`: Main Dash web application
- File browser and state selector
- Multi-view simultaneous display
- Real-time calibration loading

### `single_spectrum.py`
- `plot_spectrum()`: Quick single spectrum visualization

## Performance Notes

- Waterfall interpolation uses scipy linear interpolation (efficient)
- Filter normalization calculated from reference frequency region
- Large datasets (many cycles/states) benefit from per-filter mode
- Interactive plots use matplotlib's built-in zoom/pan

## Troubleshooting

**Missing calibration files?**
- Waterfall will use fallback calibration (-43.5 V/dB conversion)
- Power values will be less accurate but still relative

**Shifted spectra appearing?**
- Check for ADC/LO sync issues (circular buffer bug)
- Position-based filtering should exclude most problematic spectra

**S21 corrections not loading?**
- Verify highz-filterbank repo is accessible
- Check filter .s2p files exist in characterization/s_parameters/

**Memory issues with large datasets?**
- Use per-filter mode (`--filter N`) to reduce data
- Limit frequency range with `--freq-range`
- Process cycles separately
