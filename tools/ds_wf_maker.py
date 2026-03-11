import argparse
from textwrap import dedent

import os, logging
import numpy as np
from zoneinfo import ZoneInfo

from highz_exp.argparse_utils import RichHelpFormatter
from highz_exp.unit_convert import convert_utc_list_to_local
from highz_exp.file_load import DSFileLoader
from digital_spectrometer.io_utils import setup_logging
from highz_exp.spec_proc import downsample_waterfall, validate_spectra_dimensions
from digital_spectrometer.waterfall_utils import plot_waterfall_heatmap_plotly
from digital_spectrometer.params import *

MAX_PLOT_FREQ_MHZ = 300

def main_cli():
    parser = argparse.ArgumentParser(
        description="Waterfall Plotter for Digital Spectrometer Data",
                formatter_class=RichHelpFormatter,
        epilog=dedent("""
            Example Usage:
                            python tools/ds_wf_maker.py /data/20251102/ 0 --output_dir /data/20251102/waterfalls/ --segment 4 --step_f 4

            This processes state index 0 for date 20251102, saves waterfall plots to the
            specified output directory, splits the day into 4 segments, downsamples frequency
            by a factor of 4, and keeps time resolution unchanged.
        """).strip()
    )

    # --- Positional Arguments ---
    parser.add_argument("input_dir", help="Path to the directory containing date-specific data files")
    parser.add_argument(
        "state_index", type=int, help="Index of the operational state"
    )

    # --- Optional Flags ---
    parser.add_argument(
        "--output_dir", "-o",
        default=None,
        help="Directory to save output plots (default: None, to input_dir)"
    )

    parser.add_argument("--segment", "-s", type=int, default=4, help="Number of segments to split the day into for processing. Default = 4.")
    parser.add_argument("--step_f", "-f", type=int, default=4, help="Frequency downsampling step size. Default = 4, allows 0.1 MHz resolution.")
    parser.add_argument("--step_t", "-t", type=int, default=1, help="Time downsampling step size. Default = 1.")

    # Parse the arguments
    args = parser.parse_args()

    # Initialize logic
    setup_logging()

    return args

def main(date_dir, state_indx, step_f, step_t, segment, output_dir=None):
    # --- 1. Data Ingestion & Setup ---
    time_dirs = DSFileLoader.get_sorted_time_dirs(date_dir)
    normalized_date_dir = os.path.normpath(date_dir)
    date = os.path.basename(normalized_date_dir)
    if not date or not date.isdigit() or len(date) != 8:
        raise ValueError(
            f"Unable to infer date from input directory '{date_dir}'. "
            "Expected a day folder named YYYYMMDD (e.g., /path/to/20260303)."
        )
    for quartered_time_dirs in np.array_split(time_dirs, segment):
        loaded = DSFileLoader.load_and_add_timestamps(date, quartered_time_dirs, state_indx)
        timestamps, spectra, _ = DSFileLoader.read_loaded(loaded, sort='ascending', convert=True)
        
        logging.info(f"Total spectra loaded: {len(spectra)}")
        logging.info(f"Original Timezone: {timestamps[0].tzinfo}")
        
        output_dir = output_dir or date_dir
        os.makedirs(output_dir, exist_ok=True)

        # Convert to local for processing/filenames
        hwt = ZoneInfo('HST')  # Hawaii Standard Time
        local_ts = convert_utc_list_to_local(timestamps, local_timezone=hwt)
        logging.info(f"Local Time range: {local_ts[0]} to {local_ts[-1]}")

        # --- 2. Identify Daily Boundaries ---
        dates = np.array([dt.date() for dt in local_ts])
        change_indices = np.where(dates[:-1] != dates[1:])[0] + 1
        boundaries = [0] + list(change_indices) + [len(local_ts)]

        # --- 3. Process each Day ---
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            
            day_date = dates[start]
            day_ts = local_ts[start:end]
            day_spectra = spectra[start:end, :] # Slicing rows (time)
            
            logging.info(f"Processing Day {i}: {day_date} ({len(day_ts)} samples)")
            
            # Metadata for plotting
            f_mhz = faxis  # Assumed global or defined elsewhere
            freq_mask = f_mhz <= MAX_PLOT_FREQ_MHZ
            if not np.any(freq_mask):
                logging.warning(
                    f"No frequency bins <= {MAX_PLOT_FREQ_MHZ} MHz for {day_date}; skipping segment."
                )
                continue
            f_mhz = f_mhz[freq_mask]
            day_spectra = day_spectra[:, freq_mask]

            title = f"Waterfall Plot Interactive: State {state_indx}: {day_ts[0].hour:02d} - {day_ts[-1].hour:02d})"
            output_fn = f"waterfall_{state_indx}_{day_date}_{day_ts[0].hour:02d}_{day_ts[-1].hour:02d}.html"
            
            # Validation
            if not validate_spectra_dimensions(day_ts, f_mhz, day_spectra):
                logging.warning(f"Validation failed for {day_date}: {day_ts[0].hour:02d} to {day_ts[-1].hour:02d}")
                continue

            ds_ts, ds_f, ds_spec = downsample_waterfall(day_ts, f_mhz, day_spectra, step_t=step_t, step_f=step_f)
            
            plot_waterfall_heatmap_plotly(
                ds_ts, ds_spec, 
                ds_f, 
                title, 
                pjoin(output_dir, output_fn)
            )
            
if __name__ == "__main__":

    args = main_cli()

    # Call main with the parsed values
    main(
        date_dir=args.input_dir, 
        state_indx=args.state_index, 
        output_dir=args.output_dir,
        step_f=args.step_f,
        step_t=args.step_t,
        segment=args.segment
    )
