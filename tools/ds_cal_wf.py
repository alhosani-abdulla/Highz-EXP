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

import os, pathlib
import pickle
import pandas as pd
import numpy as np
from textwrap import dedent
from zoneinfo import ZoneInfo
from tqdm.auto import tqdm
import logging

from highz_exp.argparse_utils import (select_file_path, setup_cli_logging, 
    select_folder_path, prompt_int, prompt_bool)

from digital_spectrometer.waterfall_utils import plot_waterfall_heatmap_plotly
from highz_exp.sys_cal import DSCalibrationProcessor, SystemCalibrationProcessor
from highz_exp.spec_proc import downsample_waterfall
from highz_exp.spec_class import Spectrum
from highz_exp import plotter
from highz_exp.unit_convert import convert_utc_list_to_local, convert_local_to_utc
from highz_exp.load_db import get_T_data
from CAL_VARS import ND_kelvin

# ===== Editable macros =====

TEST_SITE_LATITUDE_DEG = 51.8
TEST_SITE_LONGITUDE_DEG = -176.6
TEST_SITE_ELEVATION_METERS = 5

MIN_FREQUENCY_MHZ = 25
MAX_FREQUENCY_MHZ = 215

PLOT_FREQUENCY_AXIS_STEP_MHZ = 50
PLOT_TIME_AXIS_STEP_LST_HOUR = 1

NO_SEGMENTS = 1
LOCAL_TZ = ZoneInfo("HST")

def parse_args():
    input_dir = select_folder_path(title="Select the day folder to process")
    if not input_dir:
            raise SystemExit("No input directory selected.")

    output_dir = select_folder_path(title="Select the output directory for plots")
    if not output_dir:
        raise SystemExit("No output directory selected.")

    timeline_file = select_file_path(title="Select the timeline CSV file")
    if not timeline_file:
        raise SystemExit("No timeline CSV file selected.")

    temperature_file = select_file_path(title="Select the resistor temperature CSV file")
    if not temperature_file:
        temperature_file = prompt_int("Enter resistor temperature", 300)

    segment = prompt_int("Number of segments", 4)
    vmax = prompt_int("Max value for waterfall color scale (in K)", 1000)
    fmin = prompt_int("Minimum frequency bound in MHz", MIN_FREQUENCY_MHZ)
    fmax = prompt_int("Maximum frequency bound in MHz", MAX_FREQUENCY_MHZ)
    verbose = prompt_bool("Enable verbose logging?", default=True)

    return input_dir, output_dir, timeline_file, temperature_file, segment, vmax, fmin, fmax, verbose

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

def plot_cal_ant_wf(calibrated, proc: SystemCalibrationProcessor,
    seg_indx: int, segment_output_dir, date, vmax, t_downsample=2, f_downsample=4):
    """Downsample and plot the calibrated antenna waterfall for one segment."""
    logger = logging.getLogger("ds_cal_wf")


    if t_downsample != 1 and f_downsample != 1:
        _, frequency_bin_count = calibrated['ant_T_wf'].shape
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
            calibrated['antenna_local_timestamps'][0],
            calibrated['antenna_local_timestamps'][-1],
        )

        downsampled_datetimes, downsampled_frequencies_mhz, downsampled_spectra = downsample_waterfall(
            datetimes=np.array(calibrated['antenna_local_timestamps']),
            faxis=np.array(calibrated['frequencies_mhz']),
            spectra=calibrated['ant_T_wf'],
            step_t=t_downsample,
            step_f=waterfall_frequency_step,
        )
    else:
        logger.info("[seg %d] skipping waterfall downsampling", seg_indx)
        downsampled_datetimes = np.array(calibrated['antenna_local_timestamps'])
        downsampled_frequencies_mhz = np.array(calibrated['frequencies_mhz'])
        downsampled_spectra = calibrated['ant_T_wf']

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
        calibrated['ant_T_wf'].shape,
        downsampled_spectra.shape,
    )

    return ant_temp_waterfall_path

def calibrate_loaded(proc: DSCalibrationProcessor, seg_indx: int, nd_temp_k, resistor_temp_k, local_TZ=LOCAL_TZ):
    """Run calibration and generate all per-segment plots using a preloaded processor."""
    logger = logging.getLogger("ds_cal_wf")
    logger.info("[seg %d] preparing frequency axis and medians", seg_indx)
    frequencies_mhz = proc.frequencies_mhz
    logger.info("[seg %d] computing system gain/temp from cycles", seg_indx)

    calibrator_data = proc.calibrate_per_cycle(resistor_temp_k=resistor_temp_k, nd_k=nd_temp_k)

    system_temp = [Spectrum(frequency=frequencies_mhz * 1e6, spectrum=proc.system_temp_per_cycle[i], 
        name="System Temperature") for i in range(proc.system_temp_per_cycle.shape[0])]
    system_gain = [Spectrum(frequency=frequencies_mhz * 1e6, spectrum=np.log10(proc.gain_per_cycle[i]) * 10,
        name="System Gain (dB)") for i in range(proc.gain_per_cycle.shape[0])]

    antenna_utc_timestamps = np.array(proc.raw_states["0"]["timestamps"])
    antenna_local_timestamps = convert_utc_list_to_local(antenna_utc_timestamps,
        local_timezone=LOCAL_TZ)
    segment_local_label = proc.build_segment_local_label(
        antenna_local_timestamps,
        seg_indx=seg_indx,
        timezone_name="HST",
    )
    logger.info("[seg %d] label=%s", seg_indx, segment_local_label)

    logger.info("[seg %d] building calibrated antenna waterfall", seg_indx)
    ant_T_wf = proc.calibrate_2d_state_power("0")

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
        "antenna_local_timestamps": antenna_local_timestamps,
        "frequencies_mhz": frequencies_mhz,
    }

def load_calibrator(seg_indx, data_folder, time_interval, fmin, fmax, segment) -> DSCalibrationProcessor:
    """Load data into calibrator processor."""
    proc = DSCalibrationProcessor(min_frequency_mhz=fmin,
        max_frequency_mhz=fmax,
        site_latitude_deg=TEST_SITE_LATITUDE_DEG,
        site_longitude_deg=TEST_SITE_LONGITUDE_DEG,
        site_elevation_m=TEST_SITE_ELEVATION_METERS,
    )

    states_to_load = [0, 5, 6]  # Antenna, Resistor, Noise diode
    _ = proc.load_states(data_folder, convert=False,
        no_segments=segment, seg_indx=seg_indx, 
        states_to_load=states_to_load,
        time_interval=time_interval,
    )
    
    return proc
        
def run_segment(seg_indx, data_folder, output_dir, date, timeline_info, 
                temperature_df, segment, fmin, fmax, vmax) -> list[dict] | None:
    """Run the calibration workflow for one segment index, including loading, calibration, and plotting."""
    segment_output_dir = os.path.join(output_dir, f"seg_{seg_indx}")
    os.makedirs(segment_output_dir, exist_ok=True)
    logger = logging.getLogger("ds_cal_wf")
    logger.info("[seg %d] initializing calibration processor", seg_indx)
    logger.info("[seg %d] output_dir=%s", seg_indx, segment_output_dir)

    results = []

    # iterate over timelines
    for row_idx, row in timeline_info.iterrows():
        # convert hst to utc
        start_hst = row['start_hst']
        end_hst = row['end_hst']

        proc = load_calibrator(seg_indx, data_folder, time_interval=(start_hst, end_hst), fmin=fmin, fmax=fmax, segment=segment)
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
        
        R_T, _ = get_T_data(temperature_df, proc.raw_states["5"]["timestamps"], local_tz=LOCAL_TZ)
        nd_indx = row.get("ND")
        if nd_indx is None:
            logger.info("Processing data without calibrators...")
            pass # Needs attention
        else:
            nd_indx = int(nd_indx)
            logger.info(
                "[seg %d] Using noise diode index %d from timeline for calibration", seg_indx, nd_indx)
            nd_temp = np.full_like(proc.frequencies_mhz, ND_kelvin(indx=nd_indx))
            calibrated = calibrate_loaded(
                proc=proc, seg_indx=seg_indx, nd_temp_k=nd_temp, resistor_temp_k=R_T)
            plot_cal_ant_wf(calibrated,
                proc=proc, seg_indx=seg_indx,
                segment_output_dir=segment_output_dir,
                date=date, vmax=vmax, t_downsample=2, f_downsample=4)
            results.append(calibrated)

        return results

def main():
    input_dir, output_dir, timeline_file, temperature_file, segment, vmax, fmin, fmax, verbose = parse_args()
    logger = setup_cli_logging(verbose=verbose, logger_name="ds_cal_wf")
    logger.info("Starting DS calibration workflow")

    data_folder = os.path.normpath(os.path.expanduser(input_dir))
    output_dir = os.path.expanduser(output_dir)
    date = os.path.basename(data_folder)

    if not isinstance(temperature_file, int):
        temperature_df = pd.read_csv(temperature_file)

    # Process timeline csv file to extract calibrator information (e.g., noise diode index) for each segment
    timeline_info = parse_timeline_info(timeline_file)

    if fmin >= fmax:
        raise ValueError("--fmin must be smaller than --fmax")

    segment_results = []
    segment_indices = tqdm(range(segment),
        desc="Calibrating segments", unit="seg", dynamic_ncols=True)
    
    for seg_indx in segment_indices:
        segment_indices.set_postfix_str(f"seg_{seg_indx}")
        segment_result = run_segment(seg_indx=seg_indx, data_folder=data_folder,
            output_dir=output_dir, timeline_info=timeline_info, temperature_df=temperature_df,
            date=date, fmin=fmin, fmax=fmax, vmax=vmax, segment=segment
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
                "freq_range": (fmin, fmax),
                "y_range": (0, 300)
            },
        },
        {
            "name": "sys_gain_db",
            "spectra": [spec for result in segment_results for spec in result["gain_per_cycle"]],
            "kwargs": {
                "ylabel": "Raw Power (arb.)",
                "title": f"System Gain: {date}",
                "freq_range": (fmin, fmax),
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
