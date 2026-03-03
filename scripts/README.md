# Scripts

Command-line utilities and batch runners for filterbank processing/visualization.

## Important

These scripts currently contain hardcoded absolute paths from the original environment.

Before use on a new machine, update path variables near the top of each script:

- `scripts/consolidate_all.sh`
- `scripts/batch_waterfall.sh`
- `scripts/highz_alias.sh`
- `scripts/create_animations.py`

## Prerequisites

- Run from repository root.
- Install project deps and pipenv environment.
- `ffmpeg` is required for MP4 generation in `create_animations.py`.

## Quick Start

```bash
bash scripts/consolidate_all.sh
bash scripts/batch_waterfall.sh 20251102
pipenv run python scripts/create_animations.py 20251102
```

## Script Details

### `consolidate_all.sh`

Batch consolidation across predefined days using filter calibration data.

Current behavior:

- Processes a fixed internal `DAYS` list (`11012025` to `11062025`).
- Runs `pipenv run python scripts/consolidate_filterbank_data.py ...` for each day.
- Uses hardcoded `FILTERCAL_DIR`, `OUTPUT_DIR`, and `BANDPASS_DIR`.

Usage:

```bash
bash scripts/consolidate_all.sh
```

### `batch_waterfall.sh`

Runs waterfall generation for all states, optionally for selected filter indices.

Usage:

```bash
bash scripts/batch_waterfall.sh [DATE] [FILTER_0 FILTER_1 ...]
```

Arguments:

- `DATE` (optional): Format `YYYYMMDD`; default is `20251102`.
- Filters (optional): Zero-based filter indices (`0-20`). If omitted, all filters are processed.

Current behavior:

- States processed: `0 1 2 3 4 5 6 7 1_OC`.
- Uses `tools/waterfall_plotter.py` with `--no-interactive`.
- Writes outputs under a day-specific folder derived from `DATE`.

### `create_animations.py`

Creates GIF and MP4 animations per state by stitching waterfall PNG frames across filters.

Usage:

```bash
pipenv run python scripts/create_animations.py [DATE]
```

Arguments:

- `DATE` (optional): Format `YYYYMMDD`; default is `20251102`.

Outputs:

- `state_<STATE>_waterfall_animation.gif`
- `state_<STATE>_waterfall_animation.mp4`

for each state in: `0,1,2,3,4,5,6,7,1_OC`.

### `highz_alias.sh`

Shell aliases/functions for interactive local workflows (viewing spectra, compression, plot creation).

Usage:

```bash
source scripts/highz_alias.sh
```

Then use the defined aliases/functions in your shell session.
