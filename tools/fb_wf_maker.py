import argparse
from textwrap import dedent

import os, logging
import numpy as np
from zoneinfo import ZoneInfo

from highz_exp.argparse_utils import RichHelpFormatter, setup_cli_logging
from digital_spectrometer.waterfall_utils import plot_waterfall_heatmap_plotly
from highz_exp.unit_convert import convert_utc_list_to_local
from highz_filterbank import io_utils

MAX_PLOT_FREQ_MHZ = 200

logger = logging.getLogger(__name__)


def split_by_antenna_id_changes(ts, freq, powers, antenna_ids):
    """Split arrays into contiguous chunks where antenna_ids changes.

    Parameters
    ----------
    ts : array-like
        Timestamp sequence with one entry per time sample.
    freq : array-like
        Frequency axis. If 1D, it is reused for every chunk.
        If the first dimension matches len(ts), it is split along that axis.
    powers : np.ndarray
        Power array with shape (n_times, n_freqs).
    antenna_ids : array-like
        Antenna ID per time sample (same length as ts).

    Returns
    -------
    tuple[list, list, list, list]
        (ts_chunks, freq_chunks, powers_chunks, antenna_id_chunks), where each
        item in the lists corresponds to one contiguous antenna-id block.
    """
    ts_arr = np.asarray(ts)
    freq_arr = np.asarray(freq)
    powers_arr = np.asarray(powers)
    antenna_arr = np.asarray(antenna_ids)

    if ts_arr.shape[0] != antenna_arr.shape[0]:
        raise ValueError("ts and antenna_ids must have the same length.")
    if powers_arr.shape[0] != ts_arr.shape[0]:
        raise ValueError("powers first dimension must match len(ts).")

    change_idx = np.where(antenna_arr[1:] != antenna_arr[:-1])[0] + 1
    boundaries = np.concatenate(([0], change_idx, [ts_arr.shape[0]]))

    ts_chunks = []
    freq_chunks = []
    powers_chunks = []
    antenna_id_chunks = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ts_chunks.append(ts_arr[start:end])
        powers_chunks.append(powers_arr[start:end])
        antenna_id_chunks.append(antenna_arr[start:end])

        if freq_arr.ndim >= 1 and freq_arr.shape[0] == ts_arr.shape[0]:
            freq_chunks.append(freq_arr[start:end])
        else:
            freq_chunks.append(freq_arr.copy())

    return ts_chunks, freq_chunks, powers_chunks, antenna_id_chunks

def main_cli():
    parser = argparse.ArgumentParser(
        description=(
            "Load one filterbank state for a day, optionally reuse cached calibrated data, "
            "apply frequency-window filtering, and generate an interactive waterfall plot."
        ),
        formatter_class=RichHelpFormatter,
        epilog=dedent("""
            Examples:
              1) Generate a waterfall directly from FITS files:
                 python tools/fb_wf_maker.py /data/filterbank/03102026 0 -o /data/filterbank/plots/

              2) Limit output to a frequency window (MHz):
                 python tools/fb_wf_maker.py /data/filterbank/03102026 0 --fmin 40 --fmax 180

              3) Save calibrated state data to NPZ cache for faster reruns:
                 python tools/fb_wf_maker.py /data/filterbank/03102026 0 --save

              4) Enable verbose progress logging:
                 python tools/fb_wf_maker.py /data/filterbank/03102026 0 --verbose

            Notes:
              - `state` is a numeric state index (for example 0, 1, 2).
              - If a cached NPZ exists, the script can load it instead of recalibrating.
              - Output files are written under `--output_dir` (or `input_dir` if omitted).
        """).strip()
    )

    # --- Positional Arguments ---
    parser.add_argument(
        "input_dir",
        help="Path to a day folder containing filterbank cycle directories (for example /data/filterbank/03102026).",
    )
    parser.add_argument(
        "state",
        type=int,
        help="Numeric switch-state index to load and plot.",
    )

    # --- Optional Flags ---
    parser.add_argument("--output_dir", "-o",
        default=None,
        help="Directory for output HTML/NPZ files (default: input_dir)."
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=None,
        help="Lower frequency bound in MHz; bins below this are removed.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Upper frequency bound in MHz; bins above this are removed.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save calibrated state data to NPZ cache before plotting.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO).",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Initialize logging
    setup_cli_logging(verbose=args.verbose, logger_name=__name__)

    fb_fileloader = io_utils.FBFileLoader(args.input_dir)
    output_path = os.path.join(args.output_dir if args.output_dir else args.input_dir, f"fb_calibrated_state_{args.state}.npz")
    if args.save:
        ts, freq, powers, _, antenna_ids = fb_fileloader.save(output_path=output_path, state_no=args.state, overwrite=True)
    else:
        if os.path.exists(output_path):
            user_input = input(f"Calibrated data file {output_path} already exists. Do you want to load it instead of recalibrating? (y/n): ").strip().lower()
            if user_input == 'y':
                result = fb_fileloader.load_from_npz(output_path)
                ts = result['timestamps']
                freq = result['frequencies']
                powers = result['powers']
                antenna_ids = result['antenna_ids']
                logger.info(f"Loaded calibrated data from {output_path}")
            else:
                ts, freq, powers, _, antenna_ids = fb_fileloader.load(args.state) 

    local_ts = convert_utc_list_to_local(ts, local_timezone=ZoneInfo('HST'))

    logger.info(f"Loaded data for state index {args.state} from {args.input_dir}")
    logger.info(f"Original timezone of timestamps: {ts[0].tzinfo}")

    freq_mask = np.ones(len(freq), dtype=bool)
    if args.fmin is not None:
        freq_mask &= freq >= args.fmin
    if args.fmax is not None:
        freq_mask &= freq <= args.fmax
    freq = freq[freq_mask]
    powers = powers[:, freq_mask]
    
    logger.info(f"Plotting datetime range: {local_ts[0]} to {local_ts[-1]}")
    
    # --- Spit data by antenna ID changes ---
    ts_chunks, freq_chunks, powers_chunks, antenna_id_chunks = split_by_antenna_id_changes(local_ts, freq, powers, antenna_ids)
    for i, (ts_chunk, freq_chunk, powers_chunk, antenna_id_chunk) in enumerate(zip(ts_chunks, freq_chunks, powers_chunks, antenna_id_chunks)):
        logger.info(f"Chunk {i}: Antenna ID {antenna_id_chunk[0]}, Time range {ts_chunk[0]} to {ts_chunk[-1]}, Freq range {freq_chunk[0]} to {freq_chunk[-1]} MHz")
        # --- Generate Waterfall Plot ---
        output_path = os.path.join(args.output_dir if args.output_dir else args.input_dir, f"fbwf_state_{args.state}.html")
        plot_waterfall_heatmap_plotly(ts_chunk, powers_chunk, freq_chunk,
            title=f"Filterbank Waterfall - State {args.state} - Antenna {antenna_id_chunk[0]}", output_path=output_path, vmin=-70, vmax=-10)

if __name__ == "__main__":
    args = main_cli()
