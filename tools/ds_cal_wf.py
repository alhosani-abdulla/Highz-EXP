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
import logging
import os
import numpy as np
from textwrap import dedent
from zoneinfo import ZoneInfo

from digital_spectrometer.waterfall_utils import plot_waterfall_heatmap_plotly
from digital_spectrometer.sys_cal import SystemCalibrationProcessor
from highz_exp.file_load import DSFileLoader
from highz_exp.spec_proc import downsample_waterfall
from highz_exp.spec_class import Spectrum
from highz_exp import plotter
from highz_exp.unit_convert import convert_utc_list_to_local

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
    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Calibrate one day of digital spectrometer data and generate summary outputs "
            "(median spectra, system temperature/gain, and antenna temperature waterfall)."
        ),
        formatter_class=HelpFormatter,
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
    return parser.parse_args()


def build_config(input_dir, output_dir, no_segments, vmax):
    normalized_input_dir = os.path.normpath(input_dir)
    date = os.path.basename(normalized_input_dir)
    return {
        "num_frequency_samples": NUM_FREQUENCY_SAMPLES,
        "frequency_bin_size_mhz": FREQUENCY_BIN_SIZE_MHZ,
        "test_site_latitude": TEST_SITE_LATITUDE_DEG,
        "test_site_longitude": TEST_SITE_LONGITUDE_DEG,
        "test_site_elevation_meters": TEST_SITE_ELEVATION_METERS,
        "min_f_mhz": MIN_FREQUENCY_MHZ,
        "max_f_mhz": MAX_FREQUENCY_MHZ,
        "plot_frequency_axis_step_mhz": PLOT_FREQUENCY_AXIS_STEP_MHZ,
        "plot_time_axis_step_lst_hour": PLOT_TIME_AXIS_STEP_LST_HOUR,
        "no_segments": no_segments,
        "noise_diode_temp_f1_k": NOISE_DIODE_TEMP_F1_K,
        "noise_diode_temp_f2_k": NOISE_DIODE_TEMP_F2_K,
        "noise_diode_freq_f1_mhz": NOISE_DIODE_FREQ_F1_MHZ,
        "noise_diode_freq_f2_mhz": NOISE_DIODE_FREQ_F2_MHZ,
        "resistor_temp_k": RESISTOR_TEMP_K,
        "date": date,
        "data_folder": normalized_input_dir,
        "output_dir": output_dir,
        "vmax": vmax,
    }


def build_segment_local_label(local_timestamps, seg_indx, timezone_name="HST"):
    if len(local_timestamps) == 0:
        return f"Seg {seg_indx}"
    start_local = local_timestamps[0]
    end_local = local_timestamps[-1]
    start_str = start_local.strftime("%m-%d %H:%M")
    if start_local.date() == end_local.date():
        end_str = end_local.strftime("%H:%M")
    else:
        end_str = end_local.strftime("%m-%d %H:%M")
    return f"Seg {seg_indx} {start_str}-{end_str} {timezone_name}"


def run_segment(cfg, seg_indx, logger):
    segment_output_dir = os.path.join(cfg["output_dir"], f"seg_{seg_indx}")
    os.makedirs(segment_output_dir, exist_ok=True)
    logger.info("[seg %d] Output directory: %s", seg_indx, segment_output_dir)

    logger.info("Initializing system calibration processor")
    proc = SystemCalibrationProcessor(
        num_frequency_samples=cfg["num_frequency_samples"],
        frequency_bin_size_mhz=cfg["frequency_bin_size_mhz"],
        min_frequency_mhz=cfg["min_f_mhz"],
        max_frequency_mhz=cfg["max_f_mhz"],
        site_latitude_deg=cfg["test_site_latitude"],
        site_longitude_deg=cfg["test_site_longitude"],
        site_elevation_m=cfg["test_site_elevation_meters"],
    )

    states_to_load = ["antenna", "noise_diode"]
    logger.info("Loading states: %s", states_to_load)
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

    logger.info("Preparing frequency axis and time metadata")
    frequencies_mhz = proc.prepare_frequency_axis()
    proc.slice_state_frequency_range()
    proc.compute_state_medians()
    logger.info("Frequency bins retained in range: %d", len(frequencies_mhz))

    logger.info("Loading resistor state (state 5) for system calibration")
    time_dirs = DSFileLoader.get_sorted_time_dirs(cfg["data_folder"])
    resistor_time_dirs = np.array_split(time_dirs, cfg["no_segments"])[seg_indx]
    resistor_loaded = DSFileLoader.load_and_add_timestamp(
        cfg["date"],
        list(resistor_time_dirs),
        5,
    )
    res_timestamps, res_spectra = DSFileLoader.read_loaded(
        resistor_loaded,
        sort="ascending",
        convert=False,
    )
    logger.info("Loaded resistor spectra in selected segment: %d", len(res_timestamps))
    resistor_median = np.median(
        res_spectra[:, proc.frequency_idx_range], axis=0)

    logger.info(
        "Computing system gain and system temperature from state medians")
    system_gain, system_temp = proc.calibrate_system_from_medians(
        noise_diode_temp_f1_k=cfg["noise_diode_temp_f1_k"],
        noise_diode_temp_f2_k=cfg["noise_diode_temp_f2_k"],
        noise_diode_freq_f1_mhz=cfg["noise_diode_freq_f1_mhz"],
        noise_diode_freq_f2_mhz=cfg["noise_diode_freq_f2_mhz"],
        resistor_temp_k=cfg["resistor_temp_k"],
        resistor_median=resistor_median,
    )

    logger.info(
        "Calibration outputs: system_gain=%s, system_temp=%s",
        system_gain.shape,
        system_temp.shape,
    )

    antenna_spec_median = proc.state_medians["antenna"]
    noise_diode_spec_median = proc.state_medians["noise_diode"]

    antenna_utc_timestamps = np.array(proc.raw_states["antenna"]["timestamps"])
    local_timezone = ZoneInfo("HST")
    antenna_local_timestamps = convert_utc_list_to_local(
        antenna_utc_timestamps,
        local_timezone=local_timezone,
    )
    segment_local_label = build_segment_local_label(
        antenna_local_timestamps,
        seg_indx=seg_indx,
        timezone_name="HST",
    )
    logger.info("Segment label for combined plots: %s", segment_local_label)

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
        spectrum=system_temp,
        name=f"{segment_local_label}",
    )
    system_gain_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=system_gain,
        name=f"{segment_local_label}",
    )

    sys_gain_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=np.log10(system_gain) * 10,
        name=f"{segment_local_label}",
    )

    ant_cal = proc.calibrated_temperature('antenna')
    ant_temp_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=ant_cal,
        name=f"{segment_local_label}",
    )

    logger.info("Prepared states: %s", list(proc.raw_states.keys()))

    logger.info("Building calibrated antenna waterfall")
    antenna_temperature_waterfall = proc.calibrate_2d_state_power("antenna")
    waterfall_time_step = 2
    requested_frequency_step = 2

    _, frequency_bin_count = antenna_temperature_waterfall.shape
    valid_frequency_steps = [
        divisor for divisor in range(1, requested_frequency_step + 1)
        if frequency_bin_count % divisor == 0
    ]
    waterfall_frequency_step = max(valid_frequency_steps) if valid_frequency_steps else 1
    logger.info(
        "Waterfall downsample factors selected: step_t=%d, step_f=%d",
        waterfall_time_step,
        waterfall_frequency_step,
    )

    logger.info(
        "Converted antenna timestamps to local timezone (%s): %s -> %s",
        local_timezone,
        antenna_local_timestamps[0],
        antenna_local_timestamps[-1],
    )

    downsampled_datetimes, downsampled_frequencies_mhz, downsampled_spectra = downsample_waterfall(
        datetimes=np.array(antenna_local_timestamps),
        faxis=np.array(frequencies_mhz),
        spectra=antenna_temperature_waterfall,
        step_t=waterfall_time_step,
        step_f=waterfall_frequency_step,
    )

    ant_temp_waterfall_path = os.path.join(segment_output_dir, f"{cfg['date']}_ant_cal_temp.html")
    plot_waterfall_heatmap_plotly(
        datetimes=list(downsampled_datetimes),
        spectra=downsampled_spectra,
        faxis_mhz=downsampled_frequencies_mhz,
        title=f"Antenna Calibrated Temperature: {cfg['date']}",
        unit='K',
        output_path=ant_temp_waterfall_path,
        vmin=10,
        vmax=cfg["vmax"],
        step=50,
    )

    logger.info("Saved waterfall plot: %s", ant_temp_waterfall_path)
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
        "ant_temp_spec": ant_temp_spec,
        "segment_label": segment_local_label,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("ds_cal_wf")

    logger.info("Starting DS calibration workflow")
    args = parse_args()
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    cfg = build_config(
        input_dir=input_dir,
        output_dir=output_dir,
        no_segments=args.no_segments,
        vmax=args.vmax,
    )
    logger.info("Input day directory: %s", cfg["data_folder"])
    logger.info("Output root directory: %s", cfg["output_dir"])
    logger.info("Processing all segment indices in range [0, %d]", cfg["no_segments"] - 1)
    logger.info("Frequency setup: samples=%d, bin=%.6f MHz, range=[%.1f, %.1f] MHz",
        cfg["num_frequency_samples"],
        cfg["frequency_bin_size_mhz"],
        cfg["min_f_mhz"], cfg["max_f_mhz"],
    )

    if cfg["no_segments"] <= 0:
        raise ValueError("--no-segments must be a positive integer")
    if not os.path.isdir(cfg["data_folder"]):
        raise FileNotFoundError(
            f"Data folder does not exist: {cfg['data_folder']}")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger.info("Verified input path and ensured output directory exists")

    segment_results = []
    for seg_indx in range(cfg["no_segments"]):
        logger.info("Starting segment %d/%d", seg_indx + 1, cfg["no_segments"])
        segment_result = run_segment(cfg=cfg, seg_indx=seg_indx, logger=logger)
        segment_results.append(segment_result)
        logger.info("Finished segment %d/%d", seg_indx + 1, cfg["no_segments"])

    combined_segment_count = min(3, len(segment_results))
    if combined_segment_count == 0:
        raise RuntimeError("No segment results available for combined summary plotting")

    if combined_segment_count < 3:
        logger.warning(
            "Requested combined summary from 3 segments, but only %d available",
            combined_segment_count,
        )

    combined_segments = segment_results[:combined_segment_count]
    logger.info(
        "Creating combined summary spectra plots using first %d segment(s)",
        combined_segment_count,
    )

    combined_plot_paths = {
        "cal_median": os.path.join(cfg["output_dir"], f"{cfg['date']}_cal_median_combined.png"),
        "ant_median": os.path.join(cfg["output_dir"], f"{cfg['date']}_ant_median_combined.png"),
        "sys_temp": os.path.join(cfg["output_dir"], f"{cfg['date']}_sys_temp_combined.png"),
        "sys_gain": os.path.join(cfg["output_dir"], f"{cfg['date']}_sys_gain_combined.png"),
        "sys_gain_db": os.path.join(cfg["output_dir"], f"{cfg['date']}_sys_gain_db_combined.png"),
        "ant_temp": os.path.join(cfg["output_dir"], f"{cfg['date']}_ant_temp_combined.png"),
    }

    plotter.plot_spectra(
        [spec["resistor_median_spec"] for spec in combined_segments]
        + [spec["noise_diode_median_spec"] for spec in combined_segments],
        save_path=combined_plot_paths["cal_median"],
        show_plot=False,
        ylabel="Raw Power (arb.)",
        title="Median Spectra: Resistor vs Noise Diode (Combined Segments)",
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
    )
    logger.info("Saved combined plot: %s", combined_plot_paths["cal_median"])

    plotter.plot_spectra(
        [spec["antenna_median_spec"] for spec in combined_segments],
        save_path=combined_plot_paths["ant_median"],
        show_plot=False,
        ylabel="Raw Power (arb.)",
        title="Median Spectra: Antenna (Combined Segments)",
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
    )
    logger.info("Saved combined plot: %s", combined_plot_paths["ant_median"])

    plotter.plot_spectra(
        [spec["system_temp_spec"] for spec in combined_segments],
        save_path=combined_plot_paths["sys_temp"],
        show_plot=False,
        y_range=(0, 350),
        ylabel="Temperature (K)",
        title=f"System Temperature: ({cfg['date']})",
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
        marker_freqs=(50, 100, 200),
    )
    logger.info("Saved combined plot: %s", combined_plot_paths["sys_temp"])

    plotter.plot_spectra(
        [spec["system_gain_spec"] for spec in combined_segments],
        save_path=combined_plot_paths["sys_gain"],
        show_plot=False,
        ylabel="Gain (arb.)",
        title=f"System Gain: ({cfg['date']})",
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
        marker_freqs=(50, 100, 200),
    )
    logger.info("Saved combined plot: %s", combined_plot_paths["sys_gain"])

    plotter.plot_spectra(
        [spec["sys_gain_db_spec"] for spec in combined_segments],
        y_range=(20, 60),
        ylabel='Gain (arb dB)',
        title=f"System Gain ({cfg['date']})",
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
        marker_freqs=(50, 100, 200),
        show_plot=False,
        save_path=combined_plot_paths["sys_gain_db"],
    )
    logger.info("Saved combined plot: %s", combined_plot_paths["sys_gain_db"])

    plotter.plot_spectra(
        [spec["ant_temp_spec"] for spec in combined_segments],
        y_range=(0, 1000),
        ylabel='Temperature (K)',
        title=f'Antenna: {cfg["date"]}',
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
        marker_freqs=(50, 100, 200),
        save_path=combined_plot_paths["ant_temp"],
        show_plot=False,
    )
    logger.info("Saved combined plot: %s", combined_plot_paths["ant_temp"])

    logger.info("All plots saved under segment subdirectories in: %s", cfg["output_dir"])
    logger.info("DS calibration workflow completed successfully")


if __name__ == "__main__":
    main()
