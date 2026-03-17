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
import os
import numpy as np
from textwrap import dedent
from zoneinfo import ZoneInfo
from tqdm.auto import tqdm

from highz_exp.argparse_utils import RichHelpFormatter, setup_cli_logging

from digital_spectrometer.waterfall_utils import plot_waterfall_heatmap_plotly
from highz_exp.sys_cal import DSCalibrationProcessor
from highz_exp.spec_proc import downsample_waterfall
from highz_exp.spec_class import Spectrum
from highz_exp import plotter
from highz_exp.unit_convert import convert_utc_list_to_local
from CAL_VARS import nd01_temperature_k, nd02_temperature_k

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

NO_SEGMENTS = 4

NOISE_DIODE_TEMP_F1_K = 1976
NOISE_DIODE_TEMP_F2_K = 2110
NOISE_DIODE_FREQ_F1_MHZ = 50
NOISE_DIODE_FREQ_F2_MHZ = 200
RESISTOR_TEMP_K = 273

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate one day of digital spectrometer data and generate summary outputs "
            "(median spectra, system temperature/gain, and antenna temperature waterfall)."
        ),
        formatter_class=RichHelpFormatter,
        epilog=dedent("""
            Examples:
              python tools/ds_cal_wf.py -i ~/Desktop/High-Z/Adak_2026_compressed/20260303 -o ~/Desktop/High-Z/Adak_2026_plots/0303

                            python tools/ds_cal_wf.py -i /data/20260303 -o /plots/20260303 --no-segments 4

            Notes:
              - Input directory should be a single day folder named YYYYMMDD.
                            - Segmenting splits time folders into N chunks.
                            - This script processes all segment indices from 0 to no_segments-1.
                            - Each segment output is saved under output-dir/seg_<index>/.
        """).strip(),
    )
    parser.add_argument("-i",
        "--input-dir", required=True,
        help="Path to one day folder (YYYYMMDD) containing compressed spectrometer files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for all generated PNG/HTML plots.",
    )
    parser.add_argument("-n",
        "--no-segments",
        type=int,
        default=NO_SEGMENTS,
        help="Number of equal segments used to partition day subfolders before loading.",
    )
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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level). Default logging level is WARNING.",
    )
    parser.add_argument(
        "--nd_index", default=1, type=int, choices=[1, 2], help="Noise diode index to use for calibration (1 or 2). Default is 1.",
    )
    return parser.parse_args()


def build_config(input_dir, output_dir, no_segments, vmax, fmin_mhz, fmax_mhz, nd_index):
    normalized_input_dir = os.path.normpath(input_dir)
    date = os.path.basename(normalized_input_dir)
    return {
        "num_frequency_samples": NUM_FREQUENCY_SAMPLES,
        "frequency_bin_size_mhz": FREQUENCY_BIN_SIZE_MHZ,
        "test_site_latitude": TEST_SITE_LATITUDE_DEG,
        "test_site_longitude": TEST_SITE_LONGITUDE_DEG,
        "test_site_elevation_meters": TEST_SITE_ELEVATION_METERS,
        "min_f_mhz": fmin_mhz,
        "max_f_mhz": fmax_mhz,
        "plot_frequency_axis_step_mhz": PLOT_FREQUENCY_AXIS_STEP_MHZ,
        "plot_time_axis_step_lst_hour": PLOT_TIME_AXIS_STEP_LST_HOUR,
        "no_segments": no_segments,
        "noise_diode_temp_func": nd01_temperature_k if nd_index == 1 else nd02_temperature_k,
        "resistor_temp_k": RESISTOR_TEMP_K,
        "date": date,
        "data_folder": normalized_input_dir,
        "output_dir": output_dir,
        "vmax": vmax,
    }

def calibrate_and_plot_loaded(cfg, seg_indx, logger, 
    proc, segment_output_dir, t_downsample=2, f_downsample=4):
    """Run calibration and generate all per-segment plots using a preloaded processor."""
    logger.info("[seg %d] preparing frequency axis and medians", seg_indx)
    frequencies_mhz = proc.prepare_state_medians()
    logger.info("Frequency bins retained in range: %d", len(frequencies_mhz))

    logger.info("[seg %d] computing system gain/temp from cycles", seg_indx)
    _ = proc.calibrate_system_from_cycles(resistor_temp_k=cfg["resistor_temp_k"],
        noise_diode_temp_func=cfg['noise_diode_temp_func'],
    )
    if proc.system_gain is None or proc.system_temp is None:
        raise RuntimeError("Cycle calibration did not produce system gain/temperature.")
    system_gain = proc.system_gain
    system_temp = proc.system_temp

    system_temp_med = np.median(system_temp, axis=0)
    system_gain_med = np.median(system_gain, axis=0)

    logger.info(
        "Calibration outputs: system_gain=%s, system_temp=%s",
        system_gain.shape,
        system_temp.shape,
    )

    antenna_spec_median = proc.state_medians["antenna"]
    resistor_median = proc.state_medians["resistor"]
    noise_diode_spec_median = proc.state_medians["noise_diode"]

    antenna_utc_timestamps = np.array(proc.raw_states["antenna"]["timestamps"])
    local_timezone = ZoneInfo("HST")
    antenna_local_timestamps = convert_utc_list_to_local(
        antenna_utc_timestamps,
        local_timezone=local_timezone,
    )
    segment_local_label = proc.build_segment_local_label(
        antenna_local_timestamps,
        seg_indx=seg_indx,
        timezone_name="HST",
    )
    logger.info("[seg %d] label=%s", seg_indx, segment_local_label)

    antenna_median_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=antenna_spec_median,
        name=f"{segment_local_label}",
    )

    resistor_median_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=resistor_median,
        name=f"RS | {segment_local_label}",
    )
    noise_diode_median_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=noise_diode_spec_median,
        name=f"ND | {segment_local_label}",
    )
    system_temp_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=system_temp_med,
        name=f"{segment_local_label}",
    )
    system_gain_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=system_gain_med,
        name=f"{segment_local_label}",
    )

    sys_gain_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=np.log10(system_gain_med) * 10,
        name=f"{segment_local_label} (dB)",
    )

    logger.info("[seg %d] building calibrated antenna waterfall", seg_indx)
    antenna_temperature_waterfall = proc.calibrate_2d_state_power("antenna")
    
    if t_downsample != 1 and f_downsample != 1:
        _, frequency_bin_count = antenna_temperature_waterfall.shape
        waterfall_frequency_step = proc.choose_frequency_downsample_step(
            frequency_bin_count=frequency_bin_count,
            requested_step=f_downsample,
        )
        logger.info(
            "Waterfall downsample factors selected: step_t=%d, step_f=%d",
            t_downsample, f_downsample)

        logger.info("[seg %d] local time span: %s -> %s", seg_indx, antenna_local_timestamps[0], antenna_local_timestamps[-1])

        downsampled_datetimes, downsampled_frequencies_mhz, downsampled_spectra = downsample_waterfall(
            datetimes=np.array(antenna_local_timestamps),
            faxis=np.array(frequencies_mhz),
            spectra=antenna_temperature_waterfall,
            step_t=t_downsample, step_f=waterfall_frequency_step,
        )
    else:
        logger.info("[seg %d] skipping waterfall downsampling", seg_indx)
        downsampled_datetimes = np.array(antenna_local_timestamps)
        downsampled_frequencies_mhz = np.array(frequencies_mhz)
        downsampled_spectra = antenna_temperature_waterfall

    ant_temp_waterfall_path = os.path.join(segment_output_dir, f"{cfg['date']}_ant_cal_temp.html")
    plot_waterfall_heatmap_plotly(datetimes=list(downsampled_datetimes),
        spectra=downsampled_spectra,
        faxis_mhz=downsampled_frequencies_mhz,
        title=f"Antenna Calibrated Temperature",
        unit='K', output_path=ant_temp_waterfall_path,
        vmin=10, vmax=cfg["vmax"], step=50
    )

    logger.info("[seg %d] saved waterfall=%s", seg_indx, ant_temp_waterfall_path)
    logger.info(
        "Waterfall shape: original=%s, downsampled=%s",
        antenna_temperature_waterfall.shape,
        downsampled_spectra.shape,
    )

    return {
        "resistor_median_spec": resistor_median_spec,
        "noise_diode_median_spec": noise_diode_median_spec,
        "antenna_median_spec": antenna_median_spec,
        "system_temp_spec": system_temp_spec,
        "system_gain_spec": system_gain_spec,
        "sys_gain_db_spec": sys_gain_spec,
        "segment_label": segment_local_label,
    }


def run_segment(cfg, seg_indx, logger):
    segment_output_dir = os.path.join(cfg["output_dir"], f"seg_{seg_indx}")
    os.makedirs(segment_output_dir, exist_ok=True)
    logger.info("[seg %d] output_dir=%s", seg_indx, segment_output_dir)

    logger.info("[seg %d] initializing calibration processor", seg_indx)
    proc = DSCalibrationProcessor(
        num_frequency_samples=cfg["num_frequency_samples"],
        frequency_bin_size_mhz=cfg["frequency_bin_size_mhz"],
        min_frequency_mhz=cfg["min_f_mhz"],
        max_frequency_mhz=cfg["max_f_mhz"],
        site_latitude_deg=cfg["test_site_latitude"],
        site_longitude_deg=cfg["test_site_longitude"],
        site_elevation_m=cfg["test_site_elevation_meters"],
    )

    states_to_load = ["antenna", "noise_diode", "resistor"]
    logger.info("[seg %d] loading states=%s", seg_indx, states_to_load)
    proc.load_states(
        cfg["data_folder"],
        convert=False,
        no_segments=cfg["no_segments"],
        seg_indx=seg_indx,
        states_to_load=states_to_load,
    )
    loaded_state_counts = {
        name: len(proc.raw_states[name]["timestamps"])
        for name in proc.raw_states
    }
    logger.info("[seg %d] Loaded spectra counts by state: %s", seg_indx, loaded_state_counts)

    return calibrate_and_plot_loaded(
        cfg=cfg,
        seg_indx=seg_indx,
        logger=logger,
        proc=proc,
        segment_output_dir=segment_output_dir,
    )


def main():
    args = parse_args()
    logger = setup_cli_logging(verbose=args.verbose, logger_name="ds_cal_wf")

    logger.info("Starting DS calibration workflow")
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    cfg = build_config(
        input_dir=input_dir,
        output_dir=output_dir,
        no_segments=args.no_segments,
        vmax=args.vmax,
        fmin_mhz=args.fmin,
        fmax_mhz=args.fmax,
        nd_index=args.nd_index,
    )
    logger.info("Input=%s", cfg["data_folder"])
    logger.info("Output=%s", cfg["output_dir"])
    logger.info("Segments=%d", cfg["no_segments"])
    logger.info("Frequency setup: samples=%d, bin=%.6f MHz, range=[%.1f, %.1f] MHz",
        cfg["num_frequency_samples"],
        cfg["frequency_bin_size_mhz"],
        cfg["min_f_mhz"], cfg["max_f_mhz"],
    )

    if cfg["no_segments"] <= 0:
        raise ValueError("--no-segments must be a positive integer")
    if cfg["min_f_mhz"] >= cfg["max_f_mhz"]:
        raise ValueError("--fmin must be smaller than --fmax")
    if not os.path.isdir(cfg["data_folder"]):
        raise FileNotFoundError(
            f"Data folder does not exist: {cfg['data_folder']}")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger.info("Verified input path and ensured output directory exists")

    segment_results = []
    segment_indices = tqdm(
        range(cfg["no_segments"]),
        desc="Calibrating segments",
        unit="seg",
        dynamic_ncols=True,
    )
    for seg_indx in segment_indices:
        segment_indices.set_postfix_str(f"seg_{seg_indx}")
        segment_result = run_segment(cfg=cfg, seg_indx=seg_indx, logger=logger)
        segment_results.append(segment_result)
    logger.info("Completed %d segment(s)", len(segment_results))

    combined_segment_count = min(3, len(segment_results))
    if combined_segment_count == 0:
        raise RuntimeError("No segment results available for combined summary plotting")

    if combined_segment_count < 3:
        logger.warning(
            "Requested combined summary from 3 segments, but only %d available",
            combined_segment_count,
        )

    combined_segments = segment_results[:combined_segment_count]
    logger.info("Creating combined summary plots from first %d segment(s)", combined_segment_count)

    combined_plot_paths = {
        "cal_median": os.path.join(cfg["output_dir"], f"{cfg['date']}_cal_median_combined.png"),
        "ant_median": os.path.join(cfg["output_dir"], f"{cfg['date']}_ant_median_combined.png"),
        "sys_temp": os.path.join(cfg["output_dir"], f"{cfg['date']}_sys_temp_combined.png"),
        "sys_gain": os.path.join(cfg["output_dir"], f"{cfg['date']}_sys_gain_combined.png"),
        "sys_gain_db": os.path.join(cfg["output_dir"], f"{cfg['date']}_sys_gain_db_combined.png"),
    }

    plot_jobs = [
        {
            "name": "cal_median",
            "spectra": [spec["resistor_median_spec"] for spec in combined_segments]
            + [spec["noise_diode_median_spec"] for spec in combined_segments],
            "kwargs": {
                "ylabel": "Raw Power (arb.)",
                "title": "Median Spectra: Resistor vs Noise Diode (Combined Segments)",
                "freq_range": (cfg["min_f_mhz"], cfg["max_f_mhz"]),
            },
        },
        {
            "name": "ant_median",
            "spectra": [spec["antenna_median_spec"] for spec in combined_segments],
            "kwargs": {
                "ylabel": "Raw Power (arb.)",
                "title": "Median Spectra: Antenna (Combined Segments)",
                "freq_range": (cfg["min_f_mhz"], cfg["max_f_mhz"]),
            },
        },
        {
            "name": "sys_temp",
            "spectra": [spec["system_temp_spec"] for spec in combined_segments],
            "kwargs": {
                "y_range": (0, 350),
                "ylabel": "Temperature (K)",
                "title": f"System Temperature: ({cfg['date']})",
                "freq_range": (cfg["min_f_mhz"], cfg["max_f_mhz"]),
                "marker_freqs": (50, 100, 200),
            },
        },
        {
            "name": "sys_gain",
            "spectra": [spec["system_gain_spec"] for spec in combined_segments],
            "kwargs": {
                "ylabel": "Gain (arb.)",
                "title": f"System Gain: ({cfg['date']})",
                "freq_range": (cfg["min_f_mhz"], cfg["max_f_mhz"]),
                "marker_freqs": (50, 100, 200),
            },
        },
        {
            "name": "sys_gain_db",
            "spectra": [spec["sys_gain_db_spec"] for spec in combined_segments],
            "kwargs": {
                "y_range": (20, 60),
                "ylabel": "Gain (arb dB)",
                "title": f"System Gain ({cfg['date']})",
                "freq_range": (cfg["min_f_mhz"], cfg["max_f_mhz"]),
                "marker_freqs": (50, 100, 200),
            },
        },
    ]

    for job in tqdm(plot_jobs, desc="Generating combined plots", unit="plot", dynamic_ncols=True):
        save_path = combined_plot_paths[job["name"]]
        plotter.plot_spectra(
            job["spectra"],
            save_path=save_path,
            show_plot=False,
            **job["kwargs"],
        )
        logger.info("Saved combined plot [%s]: %s", job["name"], save_path)

    logger.info("All outputs saved under: %s", cfg["output_dir"])
    logger.info("DS calibration workflow completed successfully")


if __name__ == "__main__":
    main()
