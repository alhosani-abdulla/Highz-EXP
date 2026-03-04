# Highz-EXP

Python toolkit for HighZ-EXP data loading, calibration, spectrum processing, and visualization.

## Table of contents

- [Highz-EXP](#highz-exp)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository structure](#repository-structure)
  - [Installation](#installation)
    - [Option 1: Install from GitHub (Colab or local)](#option-1-install-from-github-colab-or-local)
    - [Option 2: Editable install for development](#option-2-editable-install-for-development)
  - [Quick start](#quick-start)
  - [Core modules](#core-modules)
  - [Notebooks and tools](#notebooks-and-tools)
  - [Related repositories](#related-repositories)
  - [Troubleshooting](#troubleshooting)

## Overview

This repository provides reusable analysis code and workflow notebooks for the HighZ-EXP experiment, including:

- Spectrum and measurement abstractions
- Y-factor temperature/gain analysis
- S-parameter and reflection utilities
- Static and interactive plotting helpers

## Repository structure

- Source package: [src/highz_exp](src/highz_exp)
- Notebooks: [notebooks](notebooks) ([notebooks/README.md](notebooks/README.md))
- Workflow scripts: [scripts](scripts) ([scripts/README.md](scripts/README.md))
- Command-line tools: [tools](tools) ([tools/README.md](tools/README.md))
- Packaging/config:
	- [pyproject.toml](pyproject.toml)
	- [setup.py](setup.py)
	- [requirements.txt](requirements.txt)

## Installation

### Option 1: Install from GitHub (Colab or local)

```bash
pip install git+https://github.com/alhosani-abdulla/Highz-EXP.git
```

Use this for transient environments (for example, Colab) where editable installs are not practical.

### Option 2: Editable install for development

```bash
git clone https://github.com/alhosani-abdulla/Highz-EXP.git
cd Highz-EXP
pip install -e .
```

## Quick start

```python
import highz_exp
from highz_exp.spec_class import Spectrum
from highz_exp.fit_temperature import Y_Factor_Thermometer
from highz_exp import plotter
```

Minimal `Spectrum` example:

```python
from highz_exp.spec_class import Spectrum

spectrum = Spectrum(
		frequency=frequency_axis_hz,
		spectrum=power_values,
		name="example_spectrum",
)
```

## Core modules

Key modules in [src/highz_exp](src/highz_exp):

- [spec_class.py](src/highz_exp/spec_class.py): `Spectrum` class and spectrum utilities
- [fit_temperature.py](src/highz_exp/fit_temperature.py): Y-factor thermometer and temperature inference
- [plotter.py](src/highz_exp/plotter.py): plotting helpers and global Matplotlib defaults
- [spec_proc.py](src/highz_exp/spec_proc.py): smoothing and processing helpers
- [s_params.py](src/highz_exp/s_params.py): S-parameter utilities
- [file_load.py](src/highz_exp/file_load.py): data loading helpers

## Notebooks and tools

- Start with [notebooks/data_pipeline.ipynb](notebooks/data_pipeline.ipynb) for end-to-end processing.
- For calibration and temperature workflows, see:
	- [notebooks/Y_factor.ipynb](notebooks/Y_factor.ipynb)
	- [notebooks/amplifier_measurements.ipynb](notebooks/amplifier_measurements.ipynb)
	- [notebooks/calibrator_measurements.ipynb](notebooks/calibrator_measurements.ipynb)
- For command-line automation and utilities:
	- [scripts](scripts)
	- [tools](tools)

## Related repositories

- [adf4351-controller](https://github.com/alhosani-abdulla/adf4351-controller): Arduino control for ADF4351 PLL local oscillator workflows
- [highz-filterbank](https://github.com/alhosani-abdulla/highz-filterbank): multi-channel filterbank spectrometer pipeline

## Troubleshooting

- If imports fail, reinstall from source:

```bash
pip install --force-reinstall git+https://github.com/alhosani-abdulla/Highz-EXP.git
```

- If notebook paths fail, update the data-path setup cells in the notebook you are running.