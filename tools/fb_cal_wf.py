from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from textwrap import dedent
from typing import Any

import numpy as np
from astropy.io import fits

from highz_exp.sys_cal import SystemCalibrationProcessor
from highz_filterbank import io_utils
from highz_exp.argparse_utils import RichHelpFormatter

try:
    from rich.logging import RichHandler  # type: ignore[import-not-found]
except ImportError:
    RichHandler = None

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

    def _iter_prepared_state_rows(
            self,
            cycle_dirs: np.ndarray,
            state_filter: set[str],
            filter_exclusions_str: str | None,
        ):
        """Yield prepared spectrum rows from cycle/state FITS files.

        This intermediate handles directory traversal, state-file discovery,
        prepared-data loading/unpacking, and timestamp assignment.
        """
        for state_no, state_name in enumerate(self.state_list):
            if state_name not in state_filter:
                continue

            for cycle_dir_obj in cycle_dirs:
                cycle_dir = Path(cycle_dir_obj)
                state_path: Path | None = None

                for candidate in self._state_name_to_file_candidates(state_name, state_no):
                    candidate_path = cycle_dir / candidate
                    if candidate_path.exists():
                        state_path = candidate_path
                        break

                if state_path is None:
                    continue

                timestamps, n_spectra = self._state_file_timestamps_and_count(state_path)
                if n_spectra == 0:
                    continue

                for spectrum_idx in range(n_spectra):
                    prepared = io_utils.load_prepared_spectrum_data(
                        state_file=str(state_path),
                        spectrum_idx=spectrum_idx,
                        cycle_dir=str(cycle_dir),
                        filter_exclusions_str=filter_exclusions_str,
                    )

                    frequencies = np.asarray(prepared["frequencies"], dtype=float)
                    powers = np.asarray(prepared["powers"], dtype=float).ravel()
                    time_display = prepared["time_display"]

                    if self.total_frequencies_mhz is None:
                        self.total_frequencies_mhz = frequencies

                    if spectrum_idx < len(timestamps):
                        timestamp = timestamps[spectrum_idx]
                    else:
                        timestamp = self._parse_timestamp(time_display)

                    yield {
                        "state_name": state_name,
                        "timestamp": timestamp,
                        "frequencies": frequencies,
                        "spectrum": powers,
                        "cycle": cycle_dir.name,
                    }

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

        # Keep bin count even to match downstream assumptions used in calibration/plotting.
        if len(frequency_idx_range) % 2 != 0:
            if len(frequency_idx_range) == 1:
                raise ValueError(
                    "Selected frequency range contains only one bin; "
                    "cannot enforce an even bin count."
                )
            dropped_idx = frequency_idx_range[-1]
            logging.info(
                "Selected frequency bin count is odd (%d). Dropping highest bin %.6f MHz to make it even.",
                len(frequency_idx_range),
                self.total_frequencies_mhz[dropped_idx],
            )
            frequency_idx_range = frequency_idx_range[:-1]

        self.frequency_idx_range = frequency_idx_range
        self.frequencies_mhz = self.total_frequencies_mhz[self.frequency_idx_range]
        return self.frequencies_mhz

    def load_states(
            self,
            data_folder: str | Path,
            no_segments: int = 1,
            seg_indx: int = 0,
            states_to_load: list[str] | None = None,
            filter_exclusions_str: str | None = None,
        ) -> dict[str, dict[str, Any]]:
        """Load filterbank raw states from cycle/state FITS files and apply calibrations.

        Returns a dictionary keyed by state name with:
        - ``timestamps``: np.ndarray[datetime]
        - ``spectra``: np.ndarray[float] with shape ``(n_spectra, n_bins)``
        - ``cycles``: np.ndarray[str] cycle folder labels
        """
        if no_segments < 1:
            raise ValueError("no_segments must be >= 1.")
        if not (0 <= seg_indx < no_segments):
            raise ValueError(
                f"seg_indx must satisfy 0 <= seg_indx < no_segments ({no_segments}).")

        if states_to_load is not None:
            unknown = sorted(set(states_to_load) - set(self.state_list))
            if unknown:
                raise ValueError(f"Unknown states requested: {unknown}")
            state_filter = set(states_to_load)
        else:
            state_filter = set(self.state_list)

        data_folder = Path(data_folder)
        if not data_folder.exists():
            raise FileNotFoundError(
                f"Data folder does not exist: {data_folder}")

        # Accept both legacy "cycle_*" and current "Cycle_*" naming.
        cycle_dirs = sorted(
            [
                d
                for d in data_folder.iterdir()
                if d.is_dir() and d.name.lower().startswith("cycle_")
            ]
        )

        if len(cycle_dirs) == 0:
            raise ValueError(
                f"No cycle directories found under: {data_folder}")

        segmented_cycle_dirs = np.array_split(
            np.array(cycle_dirs, dtype=object), no_segments)[seg_indx]
        if len(segmented_cycle_dirs) == 0:
            raise ValueError(
                f"Selected segment {seg_indx} is empty for no_segments={no_segments}.")

        grouped_rows: dict[str, dict[str, list[Any]]] = {
            state_name: {"timestamps": [], "spectra": [], "cycles": []}
            for state_name in state_filter
        }
        frequency_rows: list[np.ndarray] = []

        # Avoid stale frequency-axis state when load_states() is called repeatedly.
        self.total_frequencies_mhz = None
        self.total_frequencies_2d_mhz: np.ndarray | None = None

        for row in self._iter_prepared_state_rows(
            cycle_dirs=segmented_cycle_dirs,
            state_filter=state_filter,
            filter_exclusions_str=filter_exclusions_str,
        ):
            state_name = row["state_name"]
            grouped_rows[state_name]["timestamps"].append(row["timestamp"])
            grouped_rows[state_name]["spectra"].append(row["spectrum"])
            grouped_rows[state_name]["cycles"].append(row["cycle"])
            frequency_rows.append(np.asarray(row["frequencies"], dtype=float).ravel())

        if len(frequency_rows) > 0:
            first_bin_count = frequency_rows[0].shape[0]
            for row_idx, freq_row in enumerate(frequency_rows[1:], start=1):
                if freq_row.shape[0] != first_bin_count:
                    raise ValueError(
                        "Inconsistent frequency-axis length detected across prepared spectra: "
                        f"row 0 has {first_bin_count} bins but row {row_idx} has {freq_row.shape[0]} bins."
                    )

            frequency_2d = np.vstack(frequency_rows)
            self.total_frequencies_2d_mhz = frequency_2d
            self.total_frequencies_mhz = frequency_2d[0]

            if not np.allclose(frequency_2d, self.total_frequencies_mhz[None, :], rtol=1e-10, atol=1e-12):
                mismatch_idx = np.argwhere(
                    ~np.isclose(
                        frequency_2d,
                        self.total_frequencies_mhz[None, :],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                )[0]
                bad_row = int(mismatch_idx[0])
                bad_col = int(mismatch_idx[1])
                raise ValueError(
                    "Prepared spectra do not share an identical frequency axis. "
                    f"First mismatch at row {bad_row}, bin {bad_col}: "
                    f"{frequency_2d[bad_row, bad_col]:.9f} MHz != "
                    f"{self.total_frequencies_mhz[bad_col]:.9f} MHz."
                )

        loaded: dict[str, dict[str, Any]] = {}
        for state_name in self.state_list:
            if state_name not in grouped_rows:
                continue

            state_rows = grouped_rows[state_name]
            if len(state_rows["timestamps"]) == 0:
                continue

            timestamps_arr = np.array(state_rows["timestamps"], dtype=object)
            spectra_arr = np.asarray(state_rows["spectra"])
            cycles_arr = np.array(state_rows["cycles"])

            sort_idx = np.argsort(timestamps_arr)
            loaded[state_name] = {
                "timestamps": timestamps_arr[sort_idx],
                "spectra": spectra_arr[sort_idx],
                "cycles": cycles_arr[sort_idx],
            }

        self.raw_states = loaded
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
              python tools/fb_cal_wf.py -i /data/filterbank/20260303_cycle

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
        "-n", "--no-segments", type=int, default=1, 
        help="Number of equal cycle segments to split before loading.",
    )
    parser.add_argument("--seg-indx",
        type=int, default=0, help="Zero-based segment index to load.",
    )
    parser.add_argument("--states",
        default=None, help="Optional comma-separated subset of states (e.g., antenna,resistor,noise_diode).",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=25.0,
        help="Minimum analysis frequency in MHz.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=170.0,
        help="Maximum analysis frequency in MHz.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO).",
    )
    return parser.parse_args()


def main_cli() -> None:
    args = _parse_cli_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    if RichHandler is not None:
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    logger = logging.getLogger("fb_cal_wf")

    input_dir = os.path.expanduser(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if args.no_segments < 1:
        raise ValueError("--no-segments must be >= 1")
    if not (0 <= args.seg_indx < args.no_segments):
        raise ValueError("--seg-indx must satisfy 0 <= seg-indx < no-segments")
    if args.fmin >= args.fmax:
        raise ValueError("--fmin must be smaller than --fmax")


    states_to_load = None
    if args.states:
        states_to_load = [s.strip() for s in args.states.split(",") if s.strip()]

    proc = FBCalibrationProcessor(
        min_frequency_mhz=args.fmin,
        max_frequency_mhz=args.fmax,
    )

    logger.info("Loading states from: %s", input_dir)
    proc.load_states(
        data_folder=input_dir,
        no_segments=args.no_segments,
        seg_indx=args.seg_indx,
        states_to_load=states_to_load,
    )
    selected_freqs = proc.prepare_frequency_axis()

    print(f"Loaded {len(proc.raw_states)} state(s) from {input_dir}")
    for state_name in proc.state_list:
        state_data = proc.raw_states.get(state_name)
        if state_data is None:
            continue
        print(
            f"- {state_name}: n_spectra={len(state_data['timestamps'])}, "
            f"spectra_shape={state_data['spectra'].shape}"
        )

    if proc.total_frequencies_2d_mhz is not None:
        print(f"Frequencies 2D shape: {proc.total_frequencies_2d_mhz.shape}")
    print(
        f"Selected frequency bins: {len(selected_freqs)} "
        f"(range {selected_freqs[0]:.3f}-{selected_freqs[-1]:.3f} MHz)"
    )

if __name__ == "__main__":
    main_cli()
