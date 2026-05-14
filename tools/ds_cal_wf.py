"""Digital spectrometer day-calibration workflow.

This script calibrates one day of compressed digital spectrometer data and
generates both per-segment and combined summary products.

What it does:
- Splits one `YYYYMMDD` input day into `N` time segments.
- Loads antenna and noise-diode states for each segment.
- Loads resistor state (state 5) for calibration reference.
- Computes system gain and system temperature.
- Produces calibrated antenna-temperature waterfall plots (HTML) per segment.
- Produces combined summary PNG spectra plots across the first 3 segments.
"""

import argparse
import os, pathlib
import pickle
import pandas as pd
import numpy as np
from textwrap import dedent
from zoneinfo import ZoneInfo
from tqdm.auto import tqdm
import logging

from highz_exp.argparse_utils import RichHelpFormatter, setup_cli_logging

from digital_spectrometer.waterfall_utils import plot_waterfall_heatmap_plotly
from highz_exp.sys_cal import DSCalibrationProcessor, SystemCalibrationProcessor
from highz_exp.spec_proc import downsample_waterfall
from highz_exp.spec_class import Spectrum
from highz_exp import plotter
from highz_exp.unit_convert import convert_utc_list_to_local
from highz_exp.load_db import get_T_data
from CAL_VARS import ND_temperature

# ===== Editable macros =====
NUM_FREQUENCY_SAMPLES = 16384
FREQUENCY_BIN_SIZE_MHZ = 0.025

TEST_SITE_LATITUDE_DEG = 51.8
TEST_SITE_LONGITUDE_DEG = -176.6
TEST_SITE_ELEVATION_METERS = 5

MIN_FREQUENCY_MHZ = 25
MAX_FREQUENCY_MHZ = 215

PLOT_FREQUENCY_AXIS_STEP_MHZ = 50
PLOT_TIME_AXIS_STEP_LST_HOUR = 1

NO_SEGMENTS = 1

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate one day of digital spectrometer data and generate summary outputs, including spectrum and temperature data in .fits and html formats."
            "(median spectra, system temperature/gain, and sky temperature waterfall)."
        ),
        formatter_class=RichHelpFormatter,
        epilog=dedent("""
            Examples:
                python tools/ds_cal_wf.py -i ~/Desktop/High-Z/Adak_2026_compressed/20260303 -o ~/Desktop/High-Z/Adak_2026_plots/0303
                python tools/ds_cal_wf.py -i /data/20260303 -o /plots/20260303 --no-segments 4
                python tools/ds_cal_wf.py -i /data -o /plots --fmin 30 --fmax 200 --vmax 500
            Notes:
            - Input directory should be a single day folder named YYYYMMDD.
            - Segmenting splits time folders into N chunks.
            - This script processes all segment indices from 0 to no_segments-1.
            - Each segment output is saved under output-dir/seg_<index>/.
        """).strip(),
    )
    parser.add_argument("-i",
                        "--input-dir", required=True,
                        help="""Path to one day folder (YYYYMMDD) containing compressed spectrometer files. 
        Or a parent directory containing multiple day folders (YYYYMMDD) to process in batch mode. Each day folder will be processed independently and outputs will be saved under output-dir/<day_folder>/seg_<index>/.""",
                        )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for all generated PNG/HTML plots.",
    )
    parser.add_argument("-n", "--no-segments",
                        type=int, default=NO_SEGMENTS,
                        help="Number of equal segments used to partition day subfolders before loading.",
                        )
    parser.add_argument("--timeline_file", required=True,
                        help="""Path to the timeline CSV file that contains valid observation times and information to the calibrators.""")

    parser.add_argument("--vmax",
                        type=int,
                        default=1000,
                        help="Max value for waterfall color scale (in K). Adjust based on expected antenna temperature range."
                        )
    parser.add_argument(
        "--fmin",
        type=float,
        default=MIN_FREQUENCY_MHZ,
        help="Minimum frequency bound in MHz for calibration and plotting.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=MAX_FREQUENCY_MHZ,
        help="Maximum frequency bound in MHz for calibration and plotting.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging (INFO level). Default logging level is WARNING.",
    )
    return parser.parse_args()

def parse_timeline_info(timeline_file) -> pd.DataFrame:
    """Parse the timeline CSV file to extract calibrator information."""
    if not os.path.isfile(timeline_file):
        raise FileNotFoundError(f"Timeline file not found: {timeline_file}")
    df = pd.read_csv(timeline_file)
    required_columns = {"start_hst", "end_hst"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Timeline file must contain columns: {required_columns}")

    # parse these two columns into datetime objects
    df["start_hst"] = pd.to_datetime(df["start_hst"], format="%Y-%m-%d %H:%M")
    df["end_hst"] = pd.to_datetime(df["end_hst"], format="%Y-%m-%d %H:%M")

    # make them timezone-aware in HST
    hst_tz = ZoneInfo("HST")
    df["start_hst"] = df["start_hst"].dt.tz_localize(hst_tz)
    df["end_hst"] = df["end_hst"].dt.tz_localize(hst_tz)

    return df

def plot_cal_ant_wf(proc: SystemCalibrationProcessor,
    seg_indx: int, segment_output_dir,
    date, vmax,
    frequencies_mhz,
    antenna_utc_timestamps,
    ant_T_wf,
    t_downsample=2,
    f_downsample=4,
):
    """Downsample and plot the calibrated antenna waterfall for one segment."""
    logger = logging.getLogger("ds_cal_wf")
    local_timezone = ZoneInfo("HST")
    antenna_local_timestamps = convert_utc_list_to_local(
        np.array(antenna_utc_timestamps),
        local_timezone=local_timezone,
    )

    if t_downsample != 1 and f_downsample != 1:
        _, frequency_bin_count = ant_T_wf.shape
        waterfall_frequency_step = proc.choose_frequency_downsample_step(
            frequency_bin_count=frequency_bin_count,
            requested_step=f_downsample,
        )
        logger.info(
            "Waterfall downsample factors selected: step_t=%d, step_f=%d",
            t_downsample, f_downsample,
        )
        logger.info(
            "[seg %d] local time span: %s -> %s",
            seg_indx,
            antenna_local_timestamps[0],
            antenna_local_timestamps[-1],
        )

        downsampled_datetimes, downsampled_frequencies_mhz, downsampled_spectra = downsample_waterfall(
            datetimes=np.array(antenna_local_timestamps),
            faxis=np.array(frequencies_mhz),
            spectra=ant_T_wf,
            step_t=t_downsample,
            step_f=waterfall_frequency_step,
        )
    else:
        logger.info("[seg %d] skipping waterfall downsampling", seg_indx)
        downsampled_datetimes = np.array(antenna_local_timestamps)
        downsampled_frequencies_mhz = np.array(frequencies_mhz)
        downsampled_spectra = ant_T_wf

    ant_temp_waterfall_path = os.path.join(
        segment_output_dir, f"{date}_ant_cal_temp.html"
    )
    plot_waterfall_heatmap_plotly(
        datetimes=list(downsampled_datetimes),
        spectra=downsampled_spectra,
        faxis_mhz=downsampled_frequencies_mhz,
        title="Antenna Calibrated Temperature",
        unit="K",
        output_path=ant_temp_waterfall_path,
        vmin=10,
        vmax=vmax,
        step=50,
    )

    logger.info("[seg %d] saved waterfall=%s", seg_indx, ant_temp_waterfall_path)
    logger.info(
        "Waterfall shape: original=%s, downsampled=%s",
        ant_T_wf.shape,
        downsampled_spectra.shape,
    )

    return ant_temp_waterfall_path

def calibrate_loaded(proc: SystemCalibrationProcessor, seg_indx: int, nd_temp_k, resistor_temp_k,):
    """Run calibration and generate all per-segment plots using a preloaded processor."""
    logger = logging.getLogger("ds_cal_wf")
    logger.info("[seg %d] preparing frequency axis and medians", seg_indx)
    frequencies_mhz = proc.frequencies_mhz

    logger.info("[seg %d] computing system gain/temp from cycles", seg_indx)
    calibrator_data = proc.calibrate_per_cycle(resistor_temp_k=resistor_temp_k, nd_k=nd_temp_k)
    if proc.gain_per_cycle is None or proc.system_temp_per_cycle is None:
        raise RuntimeError(
            "Cycle calibration did not produce system gain/temperature.")

    system_temp = [Spectrum(frequency=frequencies_mhz * 1e6, spectrum=proc.system_temp_per_cycle[i], 
        name="System Temperature") for i in range(proc.system_temp_per_cycle.shape[0])]
    system_gain = [Spectrum(frequency=frequencies_mhz * 1e6, spectrum=np.log10(proc.gain_per_cycle[i]) * 10,
        name="System Gain (dB)") for i in range(proc.gain_per_cycle.shape[0])]

    proc.sample_loaded_raw()

    antenna_utc_timestamps = np.array(proc.raw_states["antenna"]["timestamps"])
    local_timezone = ZoneInfo("HST")
    antenna_local_timestamps = convert_utc_list_to_local(antenna_utc_timestamps,
        local_timezone=local_timezone)
    segment_local_label = proc.build_segment_local_label(
        antenna_local_timestamps,
        seg_indx=seg_indx,
        timezone_name="HST",
    )
    logger.info("[seg %d] label=%s", seg_indx, segment_local_label)

    logger.info("[seg %d] building calibrated antenna waterfall", seg_indx)
    ant_T_wf = proc.calibrate_2d_state_power("antenna")

    ant_T_sample_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=ant_T_wf[np.random.randint(ant_T_wf.shape[0]), :],
        name=f"{segment_local_label} Sample Sky Temperature"
    )

    return {
        "segment_label": segment_local_label,
        "calibrator_data": calibrator_data,
        "gain_per_cycle": system_gain,
        "system_temp_per_cycle": system_temp,
        "ant_T_wf": ant_T_wf,
        "ant_T_sample_spec": ant_T_sample_spec,
        "antenna_utc_timestamps": antenna_utc_timestamps,
        "frequencies_mhz": frequencies_mhz,
    }

def load_calibrator(args, seg_indx, data_folder, time_interval) -> DSCalibrationProcessor:
    """Load data into calibrator processor."""
    proc = DSCalibrationProcessor(
        num_frequency_samples=NUM_FREQUENCY_SAMPLES,
        frequency_bin_size_mhz=FREQUENCY_BIN_SIZE_MHZ,
        min_frequency_mhz=args.fmin,
        max_frequency_mhz=args.fmax,
        site_latitude_deg=TEST_SITE_LATITUDE_DEG,
        site_longitude_deg=TEST_SITE_LONGITUDE_DEG,
        site_elevation_m=TEST_SITE_ELEVATION_METERS,
    )

    states_to_load = ["antenna", "noise_diode", "resistor", "blackbody"]
    _ = proc.load_states(data_folder, convert=False,
        no_segments=args.no_segments, seg_indx=seg_indx, 
        states_to_load=states_to_load,
        time_interval=time_interval,
    )
    
    return proc
        
def run_segment(args, seg_indx, data_folder, output_dir, date, timeline_info, temperature_df) -> list[dict] | None:
    """Run the calibration workflow for one segment index, including loading, calibration, and plotting."""
    segment_output_dir = os.path.join(output_dir, f"seg_{seg_indx}")
    os.makedirs(segment_output_dir, exist_ok=True)
    logger = logging.getLogger("ds_cal_wf")
    logger.info("[seg %d] initializing calibration processor", seg_indx)
    logger.info("[seg %d] output_dir=%s", seg_indx, segment_output_dir)

    results = []

    # iterate over timelines
    for row_idx, row in timeline_info.iterrows():
        start_hst = row["start_hst"]
        end_hst = row["end_hst"]

        proc = load_calibrator(args, seg_indx, data_folder, time_interval=(start_hst, end_hst))
        loaded = proc.raw_states
        if loaded is None:
            logger.info("No observation set-up in period [%s, %s] found in this segment.", start_hst, end_hst)
            continue
        else:
            loaded_state_counts = {
                name: len(proc.raw_states[name]["timestamps"])
                for name in proc.raw_states
            }
        logger.info("[seg %d] Loaded spectra counts by state for timeline row %d: %s",
                    seg_indx, row_idx, loaded_state_counts)
        
        R_T, _ = get_T_data(temperature_df, proc.raw_states["resistor"]["timestamps"])
        nd_indx = row.get("ND")
        if nd_indx is None:
            logger.info("Processing data without calibrators...")
            pass # Needs attention
        else:
            nd_indx = int(nd_indx)
            logger.info(
                "[seg %d] Using noise diode index %d from timeline for calibration", seg_indx, nd_indx)
            nd_temp = ND_temperature(proc.frequencies_mhz * 1e6, indx=nd_indx)
            calibrated = calibrate_loaded(
                proc=proc, seg_indx=seg_indx, nd_temp_k=nd_temp, resistor_temp_k=R_T)
            plot_cal_ant_wf(
                proc=proc, seg_indx=seg_indx,
                segment_output_dir=segment_output_dir,
                date=date, vmax=args.vmax,
                frequencies_mhz=calibrated["frequencies_mhz"],
                antenna_utc_timestamps=calibrated["antenna_utc_timestamps"],
                ant_T_wf=calibrated["ant_T_wf"],
                t_downsample=2, f_downsample=4,
            )
            results.append(calibrated)

        return results

def main():
    args = parse_args()
    logger = setup_cli_logging(verbose=args.verbose, logger_name="ds_cal_wf")

    logger.info("Starting DS calibration workflow")
    data_folder = os.path.normpath(os.path.expanduser(args.input_dir))
    output_dir = os.path.expanduser(args.output_dir)
    date = os.path.basename(data_folder)

    #! change this in the future
    temperature_df = pd.read_csv(pathlib.Path("~/highz2026/resistor_T_Mar.csv"))

    # Process timeline csv file to extract calibrator information (e.g., noise diode index) for each segment
    timeline_info = parse_timeline_info(args.timeline_file)

    logger.info("Input=%s", data_folder)
    logger.info("Output=%s", output_dir)
    logger.info("Segments=%d", args.no_segments)
    logger.info("Frequency setup: samples=%d, bin=%.6f MHz, range=[%.1f, %.1f] MHz",
                NUM_FREQUENCY_SAMPLES,
                FREQUENCY_BIN_SIZE_MHZ,
                args.fmin, args.fmax,
                )

    if args.no_segments <= 0:
        raise ValueError("--no-segments must be a positive integer")
    if args.fmin >= args.fmax:
        raise ValueError("--fmin must be smaller than --fmax")
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(
            f"Data folder does not exist: {data_folder}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Verified input path and ensured output directory exists")

    segment_results = []
    segment_indices = tqdm(
        range(args.no_segments),
        desc="Calibrating segments",
        unit="seg",
        dynamic_ncols=True,
    )
    for seg_indx in segment_indices:
        segment_indices.set_postfix_str(f"seg_{seg_indx}")
        segment_result = run_segment(args=args, seg_indx=seg_indx, data_folder=data_folder,
            output_dir=output_dir, timeline_info=timeline_info, temperature_df=temperature_df,
            date=date,
        )
        segment_results.extend(segment_result if segment_result is not None else [])
    
    combined_segments_path = os.path.join(output_dir, f"{date}_combined_segments.pkl")
    with open(combined_segments_path, "wb") as fh:
        pickle.dump(segment_results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved combined calibrated segments pickle: %s", combined_segments_path)

    combined_plot_paths = {
        "sys_temp": os.path.join(output_dir, f"{date}_sys_temp.png"),
        "sys_gain_db": os.path.join(output_dir, f"{date}_sys_gain_db_combined.png"),
    }

    plot_jobs = [
        {
            "name": "sys_temp",
            "spectra": [spec for result in segment_results for spec in result["system_temp_per_cycle"]],
            "kwargs": {
                "ylabel": "Raw Power",
                "title": f"System Temperature: {date}",
                "freq_range": (args.fmin, args.fmax),
                "y_range": (0, 300)
            },
        },
        {
            "name": "sys_gain_db",
            "spectra": [spec for result in segment_results for spec in result["gain_per_cycle"]],
            "kwargs": {
                "ylabel": "Raw Power (arb.)",
                "title": f"System Gain: {date}",
                "freq_range": (args.fmin, args.fmax),
                "y_range": (0, 60)
            },
        }
    ]

    for job in tqdm(plot_jobs, desc="Generating combined plots", unit="plot", dynamic_ncols=True):
        save_path = combined_plot_paths[job["name"]]
        plotter.plot_spaghetti_spectra(values=
            job["spectra"], save_path=save_path,
            show_plot=False, **job["kwargs"],
        )
        logger.info("Saved combined plot [%s]: %s", job["name"], save_path)

    logger.info("All outputs saved under: %s", output_dir)


if __name__ == "__main__":
    main()
