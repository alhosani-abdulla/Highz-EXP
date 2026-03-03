# Notebooks

Exploratory and analysis notebooks for Highz-EXP workflows.

## Prerequisites

- Open from the repository root.
- Install project dependencies:

```bash
pip install -e .
```

- Start Jupyter Lab or Notebook:

```bash
jupyter lab
```

## Notebook Index

- `data_pipeline.ipynb` — End-to-end data loading and processing workflow.
- `calibrator_measurements.ipynb` — Calibrator-focused measurements and checks.
- `calibrator_measurements copy.ipynb` — Alternate working copy of calibrator analysis.
- `amplifier_measurements.ipynb` — Amplifier measurement analysis.
- `sparam_demo.ipynb` — S-parameter exploration examples.
- `pygsm_analysis.ipynb` — PyGSM-related analysis.
- `Y_factor.ipynb` — Y-factor calculations and inspection.

## Recommended Workflow

1. Start with `data_pipeline.ipynb` to validate paths and basic data loading.
2. Move to the specialized notebooks (`calibrator_measurements.ipynb`, `amplifier_measurements.ipynb`, `Y_factor.ipynb`) for focused analysis.
3. Use `sparam_demo.ipynb` and `pygsm_analysis.ipynb` for model/system-specific studies.

## Notes

- Some notebooks may assume local data directory layouts from the original development environment.
- If a notebook fails on file paths, update path variables in the first setup/configuration cells.
