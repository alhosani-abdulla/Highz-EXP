# Tools

Command-line utilities for Highz-EXP data ingestion, calibration, and visualization.

## Directory Overview

- `ds_zipper.py`: Convert legacy DS files into packed DS format.
- `ds_wf_maker.py`: Build interactive waterfall HTML plots for one DS state.
- `ds_cal_wf.py`: Run DS day calibration and generate per-segment + combined summary plots.
- `fb_inspector.py`: Interactive filter-bank cycle inspector.

## Prerequisites

- Run commands from repository root.
- Install project in editable mode so module imports resolve:

```bash
pip install -e .
```

## Quick Start

### 1) Condense legacy DS files

```bash
python tools/ds_zipper.py /path/to/day_dir /path/to/output_root
```

### 2) Generate DS waterfall HTML plots

```bash
python tools/ds_wf_maker.py /path/to/20260303 0 --output_dir /path/to/output --segment 4 --step_f 4 --step_t 1
```

### 3) Run DS calibration workflow

```bash
python tools/ds_cal_wf.py -i /path/to/20260303 -o /path/to/output --no-segments 4 --vmax 1000
```

### 4) Open interactive filter-bank inspector

```bash
python tools/fb_inspector.py /path/to/day_dir --state 0 --filter 10
```

## Script Reference

### `ds_zipper.py`

Convert legacy DS `.npy` files (via `LegacyDSFileLoader`) into condensed DS files compatible with `DSFileLoader`.

Usage:

```bash
python tools/ds_zipper.py INPUT_DIR OUTPUT_ROOT [--pickle]
```

Arguments:

- `INPUT_DIR`: Directory containing one day of legacy DS files.
- `OUTPUT_ROOT`: Root output directory. The script writes to `OUTPUT_ROOT/<date_folder>/...`.
- `--pickle`: Save compressed outputs using pickle format instead of NumPy format.

Behavior:

- Infers `date_folder` from `INPUT_DIR` basename.
- Iterates through each hour folder and attempts condensation for DS states `0..7`.
- Logs errors per state/hour without stopping the entire run.

### `ds_wf_maker.py`

Load DS spectra for one state, split time folders into segments, optionally downsample, and render interactive waterfall HTML outputs.

Usage:

```bash
python tools/ds_wf_maker.py INPUT_DIR STATE_INDEX [--output_dir DIR] [--segment N] [--step_f N] [--step_t N]
```

Arguments:

- `INPUT_DIR`: Day directory named like `YYYYMMDD`.
- `STATE_INDEX`: State number to visualize.
- `--output_dir`, `-o`: Output directory for HTML files. Default is `INPUT_DIR`.
- `--segment`: Number of segments for splitting daily time folders. Default `4`.
- `--step_f`: Frequency downsample factor. Default `4`.
- `--step_t`: Time downsample factor. Default `1`.

Output pattern:

- `waterfall_<state>_<YYYY-MM-DD>_<start_hour>_<end_hour>.html`

### `ds_cal_wf.py`

Calibrate one DS day folder and generate both per-segment and combined summary products.

Usage:

```bash
python tools/ds_cal_wf.py -i INPUT_DAY_DIR -o OUTPUT_DIR [--no-segments N] [--vmax K]
```

Arguments:

- `-i`, `--input-dir` (required): Day directory named `YYYYMMDD` containing compressed DS files.
- `-o`, `--output-dir` (required): Root output directory.
- `-n`, `--no-segments`: Number of equal time-directory segments. Default `4`.
- `--vmax`: Upper colorbar limit (Kelvin) for antenna calibrated waterfall plots. Default `1000`.

Pipeline summary:

- Loads `antenna` and `noise_diode` states per segment.
- Loads resistor `state 5` separately per segment for calibration reference.
- Computes system gain and system temperature.
- Saves one calibrated antenna-temperature waterfall HTML per segment.
- Saves combined PNG summary spectra from the first 3 available segments.

Outputs:

- Per segment:
  - `OUTPUT_DIR/seg_<idx>/<DATE>_ant_cal_temp.html`
- Combined plots in `OUTPUT_DIR`:
  - `<DATE>_cal_median_combined.png`
  - `<DATE>_ant_median_combined.png`
  - `<DATE>_sys_temp_combined.png`
  - `<DATE>_sys_gain_combined.png`
  - `<DATE>_sys_gain_db_combined.png`
  - `<DATE>_ant_temp_combined.png`

Configuration:

- Frequency bounds, site coordinates, and calibration constants are set in the editable macros at the top of `tools/ds_cal_wf.py`.

### `fb_inspector.py`

Interactive viewer for filter-bank spectra by cycle using `SpectrumInspector`.

Usage:

```bash
python tools/fb_inspector.py DAY_DIR --state STATE [--filter N] [--reference-spectrum CYCLE:IDX]
```

Arguments:

- `DAY_DIR`: Path to a day directory (for example `20251102`).
- `--state` (required): State identifier to inspect.
- `--filter`: Filter channel index (`0-20`), default `10`.
- `--reference-spectrum`: Optional reference spectrum in `cycle_name:index` format.

Navigation:

- Use left/right arrows or UI buttons to change cycle.
- Press `Q` to quit.

## Troubleshooting

- If imports fail, verify you ran `pip install -e .` from repo root.
- If a day path fails, check the folder exists and follows expected `YYYYMMDD` naming.
- If no calibration output appears, confirm input contains required states (`antenna`, `noise_diode`, resistor `state 5`).
