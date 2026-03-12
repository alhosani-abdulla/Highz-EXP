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
        ts, freq, powers, cycle_ids = fb_fileloader.save(output_path=output_path, state_no=args.state, overwrite=True)
    else:
        if os.path.exists(output_path):
            user_input = input(f"Calibrated data file {output_path} already exists. Do you want to load it instead of recalibrating? (y/n): ").strip().lower()
            if user_input == 'y':
                result = fb_fileloader.load_from_npz(output_path)
                ts = result['timestamps']
                freq = result['frequencies']
                powers = result['powers']
                cycle_ids = result['cycle_ids']
                logger.info(f"Loaded calibrated data from {output_path}")
            else:
                ts, freq, powers, cycle_ids = fb_fileloader.load(args.state)

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
    
    # --- Generate Waterfall Plot ---
    output_path = os.path.join(args.output_dir if args.output_dir else args.input_dir, f"fbwf_state_{args.state}.html")
    plot_waterfall_heatmap_plotly(local_ts, powers, freq,
        title=f"Filterbank Waterfall - State {args.state}", output_path=output_path, vmin=-70, vmax=-10)

if __name__ == "__main__":
    args = main_cli()
