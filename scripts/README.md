# Filterbank Spectrometer Scripts

Command-line scripts and utilities for working with High-Z filterbank spectrometer data.

## Quick Start

### Consolidate Raw Data
```bash
bash consolidate_all.sh 11042025
```

### Create Waterfall Plots
```bash
bash batch_waterfall.sh 20251102
```

### Create Animations
```bash
pipenv run python create_animations.py 20251102
```

## Scripts

### consolidate_all.sh
Consolidate all days from the raw Bandpass directory with automatic filter calibration matching.

Usage: `./consolidate_all.sh [DATE]`
- DATE: Format YYYYMMDD (e.g., 20251102)

### batch_waterfall.sh
Create waterfall plots for all filters and states for a given date.

Usage: `./batch_waterfall.sh [DATE] [filters_to_process]`
- DATE: Format YYYYMMDD
- Filters: Optional, space-separated filter numbers (0-20)

### create_animations.py
Create animated GIFs and MP4 videos from waterfall plots.

Usage: `python create_animations.py [DATE]`
- DATE: Format YYYYMMDD

## Examples Directory

The `examples/` subdirectory contains example usage scripts and diagnostics.
