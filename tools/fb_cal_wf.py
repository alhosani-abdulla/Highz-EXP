from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
from astropy.io import fits

from highz_exp.sys_cal import SystemCalibrationProcessor
from highz_filterbank import io_utils
from highz_exp.argparse_utils import RichHelpFormatter, setup_cli_logging
from highz_exp import plotter
from CAL_VARS import nd01_temperature_k
from ds_cal_wf import calibrate_and_plot_loaded


RESISTOR_TEMP_K = 273


class FBCalibrationProcessor(SystemCalibrationProcessor):
    """Filterbank calibration loader for consolidated cycle/state FITS data.

    Expected layout under ``data_folder``:
    - ``cycle_*/state_0.fits``, ``cycle_*/state_1.fits``, ...
    - each state FITS contains a table with one row per measurement/spectrum
    """

    total_frequencies_2d_mhz: np.ndarray | None = None

    @staticmethod
    def _parse_timestamp(raw_timestamp: Any) -> datetime:
        """Parse supported timestamp formats into UTC datetime objects."""
        if isinstance(raw_timestamp, datetime):
            return raw_timestamp

        ts = str(raw_timestamp).strip().replace(".fits", "")

        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m%d%Y_%H%M%S"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue

        # Keep behavior permissive for custom ISO-like strings.
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported timestamp format: '{raw_timestamp}'") from exc

    @staticmethod
    def _state_name_to_file_candidates(state_name: str, state_no: int) -> list[str]:
        """Map canonical state name to expected filterbank filename(s)."""
        if state_name == "open_circuit":
            # Consolidated filterbank data may use state_1_OC.fits for OC calibration.
            return ["state_1_OC.fits", f"state_{state_no}.fits"]
        return [f"state_{state_no}.fits"]

    @staticmethod
    def _state_file_timestamps_and_count(state_path: Path) -> tuple[list[datetime], int]:
        """Read timestamp column and row count from one state FITS file."""
        timestamps: list[datetime] = []

        with fits.open(state_path) as hdul:
            if len(hdul) < 2 or hdul[1].data is None:
                return timestamps, 0

            table = hdul[1].data
            names = set(table.names or [])

            if "SPECTRUM_TIMESTAMP" in names:
                timestamps = [
                    FBCalibrationProcessor._parse_timestamp(
                        row["SPECTRUM_TIMESTAMP"])
                    for row in table
                ]
            elif "TIME_RPI2" in names:
                timestamps = [
                    FBCalibrationProcessor._parse_timestamp(row["TIME_RPI2"])
                    for row in table
                ]

            return timestamps, len(table)
        
    def prepare_frequency_axis(self) -> np.ndarray:
        """Create filterbank frequency axis from loaded/prepared frequencies."""
        if self.total_frequencies_mhz is None:
            raise ValueError(
                "Frequency axis is not available yet. "
                "Call load_states() first so prepared frequencies can be captured."
            )

        frequency_idx_range = np.where(
            (self.total_frequencies_mhz >= self.min_frequency_mhz)
            & (self.total_frequencies_mhz <= self.max_frequency_mhz)
        )[0]

        if len(frequency_idx_range) == 0:
            raise ValueError(
                "No frequency bins found in selected range "
                f"[{self.min_frequency_mhz}, {self.max_frequency_mhz}] MHz."
            )

        self.frequency_idx_range = frequency_idx_range
        self.frequencies_mhz = self.total_frequencies_mhz[self.frequency_idx_range]
        return self.frequencies_mhz

    def load_states(
        self,
        data_folder: str | Path,
        states_to_load: list[str] | None = None,
        convert=True
    ) -> dict[str, dict[str, Any]]:
        """Load filterbank raw states from cycle/state FITS files and apply calibrations.

        Returns a dictionary keyed by state name with:
        - ``timestamps``: np.ndarray[datetime]
        - ``spectra``: np.ndarray[float] with shape ``(n_spectra, n_bins)``
        - ``cycles``: np.ndarray[str] cycle folder labels
        - ``convert``: bool whether spectra were converted from log to linear units
        """
        data_folder = Path(data_folder)
        date = data_folder.name
        if not date or not date.isdigit() or len(date) != 8:
            raise ValueError(
                f"Unable to infer date from input directory '{data_folder}'. "
                "Expected a day folder named MMDDYYYY (e.g., /path/to/03082026)."
            )
        if not data_folder.exists():
            raise FileNotFoundError(
                f"Data folder does not exist: {data_folder}")

        if states_to_load is not None:
            unknown = sorted(set(states_to_load) - set(self.state_list))
            if unknown:
                raise ValueError(f"Unknown states requested: {unknown}")
            state_filter = set(states_to_load)
        else:
            state_filter = set(self.state_list)

        loaded = {}
        loaded: dict[str, dict[str, Any]] = {}
        
        fl = io_utils.FBFileLoader(data_folder)
        for state_no, name in enumerate(self.state_list):
            if name not in state_filter:
                continue
            timestamps, frequency, spectra, cycles = fl.load(state_no=state_no)
            if convert:
                spectra = 10 ** (spectra / 10)  # Convert from dB to linear units if needed.

            loaded[name] = {"timestamps": timestamps, "frequencies": frequency,
                            "spectra": spectra, "cycles": cycles, "convert": convert}

        self.raw_states = loaded
        self.total_frequencies_mhz = frequency
        return loaded

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load filterbank cycle/state FITS data and validate prepared frequency-axis "
            "consistency across all loaded spectra."
        ),
        formatter_class=RichHelpFormatter,
        epilog=dedent("""
            Example:
                            python tools/fb_cal_wf.py -i /data/filterbank/20260303_cycle -o /plots/20260303

            Notes:
              - Input directory should contain cycle folders (cycle_* or Cycle_*).
              - Use --states antenna,resistor to load a subset of switch states.
        """).strip(),
    )
    parser.add_argument(
        "-i", "--input-dir", required=True,
        help="Path to the consolidated filterbank folder containing cycle_* directories.",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Output directory for generated plots.",
    )
    
    parser.add_argument("-l", "--load-states", action="store_true",
                        help="Load states from pickle file if available, instead of re-processing FITS files.")

    parser.add_argument(
        "--fmin",type=float,
        default=25.0, help="Minimum analysis frequency in MHz.",
    ) 
    parser.add_argument(
        "--fmax", type=float,
        default=170, help="Maximum analysis frequency in MHz.",
    )
    parser.add_argument(
        "--vmax", type=int,
        default=1000, help="Maximum value for waterfall color scale in Kelvin.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO).",
    )
    return parser.parse_args()


def build_config(
    input_dir: str,
    output_dir: str,
    fmin_mhz: float,
    fmax_mhz: float,
    vmax: float,
):
    normalized_input_dir = os.path.normpath(input_dir)
    date = os.path.basename(normalized_input_dir)
    return {
        "min_f_mhz": fmin_mhz,
        "max_f_mhz": fmax_mhz,
        "date": date,
        "data_folder": normalized_input_dir,
        "output_dir": output_dir,
        "vmax": vmax,
        "noise_diode_temp_func": nd01_temperature_k,
        "resistor_temp_k": RESISTOR_TEMP_K,
    }


def main_cli() -> None:
    args = _parse_cli_args()

    logger = setup_cli_logging(verbose=args.verbose, logger_name="fb_cal_wf")

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if args.fmin >= args.fmax:
        raise ValueError("--fmin must be smaller than --fmax")
    if args.vmax <= 0:
        raise ValueError("--vmax must be positive")

    os.makedirs(output_dir, exist_ok=True)

    pickle_path = os.path.join(output_dir, f"{os.path.basename(input_dir)}_fb_calibration.pkl")
    if args.load_states and os.path.isfile(pickle_path):
        logger.info("Loading states from pickle file: %s", pickle_path)
        proc = FBCalibrationProcessor.load_pickle(pickle_path)
    else:
        logger.warning("No pickle file found at %s. Loading states from FITS files instead.", pickle_path)
        
        cfg = build_config(
            input_dir=input_dir,
            output_dir=output_dir,
            fmin_mhz=args.fmin,
            fmax_mhz=args.fmax,
            vmax=args.vmax,
        )

        states_to_load = ["antenna", "noise_diode", "resistor"]

        proc = FBCalibrationProcessor(min_frequency_mhz=args.fmin, max_frequency_mhz=args.fmax)

        logger.info("Loading states from: %s", input_dir)
        proc.load_states(
            data_folder=input_dir,
            states_to_load=states_to_load,
        )
        
        proc.save_pickle(pickle_path)
        logger.info("Saved loaded states to pickle file: %s", pickle_path)

    segment_result = calibrate_and_plot_loaded(
        cfg=cfg, seg_indx=0,
        logger=logger, proc=proc,
        segment_output_dir=output_dir,
        t_downsample=1, f_downsample=1
    )

    # Plot selected spectra returned from calibration helper.
    spectra_plot_paths = {
        "cal_median": os.path.join(output_dir, f"{cfg['date']}_fb_cal_median.png"),
        "sys_temp": os.path.join(output_dir, f"{cfg['date']}_fb_sys_temp.png"),
        "sys_gain": os.path.join(output_dir, f"{cfg['date']}_fb_sys_gain.png"),
    }
    plot_jobs = [
        {
            "name": "cal_median",
            "spectra": [
                segment_result["resistor_median_spec"],
                segment_result["noise_diode_median_spec"],
                segment_result["antenna_median_spec"],
            ],
            "kwargs": {
                "ylabel": "Raw Power (arb.)",
                "title": f"Median Spectra ({cfg['date']})",
                "freq_range": (cfg["min_f_mhz"], cfg["max_f_mhz"]),
            },
        },
        {
            "name": "sys_temp",
            "spectra": [segment_result["system_temp_spec"]],
            "kwargs": {
                "ylabel": "Temperature (K)",
                "title": f"System Temperature ({cfg['date']})",
                "freq_range": (cfg["min_f_mhz"], cfg["max_f_mhz"]),
                "marker_freqs": (50, 100, 200),
            },
        },
        {
            "name": "sys_gain",
            "spectra": [segment_result["system_gain_spec"], segment_result["sys_gain_db_spec"]],
            "kwargs": {
                "ylabel": "Gain (arb. / dB)",
                "title": f"System Gain ({cfg['date']})",
                "freq_range": (cfg["min_f_mhz"], cfg["max_f_mhz"]),
                "marker_freqs": (50, 100, 200),
            },
        },
    ]

    for job in plot_jobs:
        save_path = spectra_plot_paths[job["name"]]
        plotter.plot_spectra(
            job["spectra"],
            save_path=save_path,
            show_plot=False,
            **job["kwargs"],
        )
        logger.info("Saved FB spectra plot [%s]: %s", job["name"], save_path)
 

if __name__ == "__main__":
    main_cli()
