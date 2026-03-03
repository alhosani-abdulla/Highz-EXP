# Tools

Small command-line utilities for Highz-EXP data handling and visualization.

## Prerequisites

- Run commands from the repository root.
- Install the package in editable mode so imports resolve correctly:

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
python tools/ds_wf_maker.py /path/to/day_dir 0 --output_dir /path/to/output --segment 4 --step_f 4 --step_t 1
```

### 3) Open interactive cycle/state inspector for Filter Bank.

```bash
python tools/fb_inspector.py /path/to/day_dir --state 0 --filter 10
```

## Scripts

### `ds_zipper.py`

Condenses legacy DS `.npy` files (loaded with `LegacyDSFileLoader`) into the newer packed format (readable by `DSFileLoader`) to reduce file clutter.

Usage:

```bash
python tools/ds_zipper.py INPUT_DIR OUTPUT_ROOT [--pickle]
```

Arguments:

- `INPUT_DIR`: Directory containing one day of legacy files.
- `OUTPUT_ROOT`: Root output directory. The script creates `OUTPUT_ROOT/<date_folder>/...` automatically.
- `--pickle`: Save with pickle format instead of NumPy format.

Notes:

- The date folder is inferred from the basename of `INPUT_DIR`.
- Processing is attempted per hour and per state (`0..7`), with errors logged per state.

---

### `ds_wf_maker.py`

Loads digital spectrometer data for a state, optionally downsamples in time/frequency, splits processing into segments, and writes interactive waterfall HTML files.

Usage:

```bash
python tools/ds_wf_maker.py INPUT_DIR STATE_INDEX [--output_dir DIR] [--segment N] [--step_f N] [--step_t N]
```

Arguments:

- `INPUT_DIR`: Date directory containing time-sliced DS data.
- `STATE_INDEX`: Operational state index to plot.
- `--output_dir, -o`: Output directory for generated HTML files (default: input directory).
- `--segment`: Number of segments used when splitting daily processing (default: `4`).
- `--step_f`: Frequency downsampling step (default: `4`, equivalent to 0.1 MHz resolution in current workflow).
- `--step_t`: Time downsampling step (default: `1`).

Outputs:

- One or more files like:
	- `waterfall_<state>_<YYYY-MM-DD>_<start_hour>_<end_hour>.html`

---

### `interactive_inspector.py`

Interactive viewer for state spectra across cycles using `SpectrumInspector`.

Usage:

```bash
python tools/interactive_inspector.py DAY_DIR --state STATE [--filter N] [--reference-spectrum CYCLE:IDX]
```

Arguments:

- `DAY_DIR`: Path to a day directory (example: `20251102`).
- `--state`: Required state identifier.
- `--filter`: Filter channel index (`0-20`, default: `10`).
- `--reference-spectrum`: Optional reference in the format `cycle_name:index`.

Navigation:

- Use left/right arrow keys or UI buttons to move between cycles.
- Press `Q` to quit.

## Troubleshooting

- If imports fail, confirm you are at repo root and ran `pip install -e .`.
- If a path error occurs, verify the day directory exists and matches expected structure.
