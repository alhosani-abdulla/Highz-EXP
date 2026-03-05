import argparse
import logging
import os
import numpy as np
from textwrap import dedent

from digital_spectrometer.waterfall_utils import plot_waterfall_heatmap_plotly
from digital_spectrometer.sys_cal import SystemCalibrationProcessor
from highz_exp.file_load import DSFileLoader
from highz_exp.spec_proc import downsample_waterfall
from highz_exp.spec_class import Spectrum
from highz_exp import plotter
from scipy.constants import Boltzmann as kB
from scipy.constants import c, epsilon_0 as e0

eta = 377
RL = 50

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
SEG_INDX = 0

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

              python tools/ds_cal_wf.py -i /data/20260303 -o /plots/20260303 --no-segments 4 --seg-indx 1

            Notes:
              - Input directory should be a single day folder named YYYYMMDD.
              - Segmenting splits time folders into N chunks and loads only one chunk.
              - seg_indx is zero-based (valid range: 0 to no_segments-1).
        """).strip(),
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
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
    parser.add_argument("-s",
        "--seg-indx",
        type=int,
        default=SEG_INDX,
        help="Zero-based segment index to load from the partitioned day folder.",
    )
    return parser.parse_args()


def build_config(input_dir, output_dir, no_segments, seg_indx):
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
        "seg_indx": seg_indx,
        "noise_diode_temp_f1_k": NOISE_DIODE_TEMP_F1_K,
        "noise_diode_temp_f2_k": NOISE_DIODE_TEMP_F2_K,
        "noise_diode_freq_f1_mhz": NOISE_DIODE_FREQ_F1_MHZ,
        "noise_diode_freq_f2_mhz": NOISE_DIODE_FREQ_F2_MHZ,
        "resistor_temp_k": RESISTOR_TEMP_K,
        "date": date,
        "data_folder": normalized_input_dir,
        "output_dir": output_dir,
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
        seg_indx=args.seg_indx,
    )
    logger.info("Input day directory: %s", cfg["data_folder"])
    logger.info("Output directory: %s", cfg["output_dir"])
    logger.info(
        "Segment selection: no_segments=%d, seg_indx=%d",
        cfg["no_segments"],
        cfg["seg_indx"],
    )
    logger.info(
        "Frequency setup: samples=%d, bin=%.6f MHz, range=[%.1f, %.1f] MHz",
        cfg["num_frequency_samples"],
        cfg["frequency_bin_size_mhz"],
        cfg["min_f_mhz"],
        cfg["max_f_mhz"],
    )

    if not os.path.isdir(cfg["data_folder"]):
        raise FileNotFoundError(
            f"Data folder does not exist: {cfg['data_folder']}")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger.info("Verified input path and ensured output directory exists")

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
        seg_indx=cfg["seg_indx"],
        states_to_load=states_to_load,
    )
    loaded_state_counts = {
        name: len(proc.raw_states[name]["timestamps"])
        for name in proc.raw_states
    }
    logger.info("Loaded spectra counts by state: %s", loaded_state_counts)

    logger.info("Preparing frequency axis and time metadata")
    frequencies_mhz = proc.prepare_frequency_axis()
    proc.slice_state_frequency_range()
    proc.compute_sidereal_timestamps()
    time_ticks = proc.compute_time_ticks_by_state(
        step_hour=cfg["plot_time_axis_step_lst_hour"])
    proc.compute_state_medians()
    logger.info("Frequency bins retained in range: %d", len(frequencies_mhz))

    logger.info("Loading resistor state (state 5) for system calibration")
    time_dirs = DSFileLoader.get_sorted_time_dirs(cfg["data_folder"])
    resistor_time_dirs = np.array_split(time_dirs, cfg["no_segments"])[cfg["seg_indx"]]
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

    nd_temp = proc.nd_temp
    logger.info(
        "Calibration outputs: system_gain=%s, system_temp=%s, nd_temp=%s",
        system_gain.shape,
        system_temp.shape,
        nd_temp.shape,
    )

    plot_frequency_tick_idx, plot_frequency_tick_labels = proc.frequency_ticks(
        frequencies_mhz,
        step_mhz=cfg["plot_frequency_axis_step_mhz"],
    )
    logger.info(
        "Computed plot ticks every %.1f MHz (%d ticks)",
        cfg["plot_frequency_axis_step_mhz"],
        len(plot_frequency_tick_idx),
    )

    state_list = list(proc.raw_states.keys())
    for name in state_list:
        globals()[f"{name}_timestamps"] = proc.raw_states[name]["timestamps"]
        globals()[f"{name}_spectra"] = proc.raw_states[name]["spectra"]
        globals()[f"{name}_power"] = proc.state_power[name]
        globals()[f"{name}_sidereal_ts"] = proc.sidereal_time[name]
        globals()[f"{name}_spec_median"] = proc.state_medians[name]
        globals()[f"{name}_plot_ts_tick_idx"] = time_ticks[name]["idx"]
        globals()[f"{name}_plot_ts_tick_labels"] = time_ticks[name]["labels"]

    antenna_spec_median = proc.state_medians["antenna"]
    noise_diode_spec_median = proc.state_medians["noise_diode"]

    antenna_median_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=antenna_spec_median,
        name="Antenna Median",
    )

    resistor_median_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=resistor_median,
        name="Resistor Median",
    )
    noise_diode_median_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=noise_diode_spec_median,
        name="Noise Diode Median",
    )
    system_temp_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=system_temp,
        name="System Temperature",
    )
    system_gain_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=system_gain,
        name="System Gain",
    )

    plot_paths = {
        "cal_median": os.path.join(cfg["output_dir"], f"{cfg['date']}_cal_median.png"),
        "ant_median": os.path.join(cfg["output_dir"], f"{cfg['date']}_ant_median.png"),
        "sys_temp": os.path.join(cfg["output_dir"], f"{cfg['date']}_sys_temp.png"),
        "sys_gain": os.path.join(cfg["output_dir"], f"{cfg['date']}_sys_gain.png"),
        "sys_gain_db": os.path.join(cfg["output_dir"], f"{cfg['date']}_sys_gain_db.png"),
        "ant_temp": os.path.join(cfg["output_dir"], f"{cfg['date']}_ant_temp.png"),
    }

    plotter.plot_spectra(
        [resistor_median_spec, noise_diode_median_spec],
        save_path=plot_paths["cal_median"],
        show_plot=False,
        ylabel="Raw Power (arb.)",
        title="Median Spectra: Resistor vs Noise Diode",
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
    )
    logger.info("Saved plot: %s", plot_paths["cal_median"])

    plotter.plot_spectra(
        [antenna_median_spec],
        save_path=plot_paths["ant_median"],
        show_plot=False,
        ylabel="Raw Power (arb.)",
        title="Median Spectra: Antenna",
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
    )
    logger.info("Saved plot: %s", plot_paths["ant_median"])

    plotter.plot_spectra(
        [system_temp_spec],
        save_path=plot_paths["sys_temp"],
        show_plot=False,
        y_range=(0, 350),
        ylabel="Temperature (K)",
        title=f"System Temperature: ({cfg['date']})",
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
        marker_freqs=(50, 100, 200),
    )
    logger.info("Saved plot: %s", plot_paths["sys_temp"])

    plotter.plot_spectra(
        [system_gain_spec],
        save_path=plot_paths["sys_gain"],
        show_plot=False,
        ylabel="Gain (arb.)",
        title=f"System Gain: ({cfg['date']})",
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
        marker_freqs=(50, 100, 200),
    )
    logger.info("Saved plot: %s", plot_paths["sys_gain"])

    sys_gain_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=np.log10(system_gain) * 10,
        name='System Gain',
    )

    plotter.plot_spectra(
        [sys_gain_spec], y_range=(20, 60),
        ylabel='Gain (arb dB)',
        title=f"System Gain ({cfg['date']})",
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
        marker_freqs=(50, 100, 200),
        show_plot=False,
        save_path=plot_paths["sys_gain_db"],
    )
    logger.info("Saved plot: %s", plot_paths["sys_gain_db"])

    ant_cal = proc.calibrated_temperature('antenna')
    ant_temp_spec = Spectrum(
        frequency=frequencies_mhz * 1e6,
        spectrum=ant_cal,
        name='Antenna Temperature',
    )

    plotter.plot_spectra(
        [ant_temp_spec], y_range=(0, 1000),
        ylabel='Temperature (K)',
        title=f'Antenna: {cfg["date"]}',
        freq_range=(cfg["min_f_mhz"], cfg["max_f_mhz"]),
        marker_freqs=(50, 100, 200),
        save_path=plot_paths["ant_temp"],
        show_plot=False,
    )
    logger.info("Saved plot: %s", plot_paths["ant_temp"])

    logger.info("Prepared states: %s", state_list)
    logger.info("Frequency tick labels: %s", plot_frequency_tick_labels)

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

    antenna_timestamps = np.array(proc.raw_states["antenna"]["timestamps"])
    downsampled_datetimes, downsampled_frequencies_mhz, downsampled_spectra = downsample_waterfall(
        datetimes=antenna_timestamps,
        faxis=np.array(frequencies_mhz),
        spectra=antenna_temperature_waterfall,
        step_t=waterfall_time_step,
        step_f=waterfall_frequency_step,
    )

    plot_paths["ant_temp_waterfall"] = os.path.join(cfg["output_dir"], f"{cfg['date']}_ant_cal_temp.html")
    plot_waterfall_heatmap_plotly(
        datetimes=list(downsampled_datetimes),
        spectra=downsampled_spectra,
        faxis_mhz=downsampled_frequencies_mhz,
        title=f"Antenna Calibrated Temperature: {cfg['date']}",
        cbar_title='Temperature (K)',
        output_path=plot_paths["ant_temp_waterfall"],
        vmin=0,
        vmax=1000,
    )

    logger.info("Saved waterfall plot: %s", plot_paths["ant_temp_waterfall"])
    logger.info(
        "Waterfall shape: original=%s, downsampled=%s",
        antenna_temperature_waterfall.shape,
        downsampled_spectra.shape,
    )
    logger.info("All plots saved to: %s", cfg["output_dir"])
    logger.info("DS calibration workflow completed successfully")


if __name__ == "__main__":
    main()
