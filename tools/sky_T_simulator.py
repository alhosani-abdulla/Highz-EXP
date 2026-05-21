# The code in this file does what notebooks/pygsm_analysis.ipynb does, but in a script form. This is to make it easier to 
# run the code and save the output for further analysis.
# Basically, this script generates simulated sky temperature data based on the FEKO gain patterns and GSM sky maps, and saves the output in a pickle file for further analysis.
import numpy as np
import pandas as pd
import datetime
import pathlib
import pickle, argparse

from highz_exp.argparse_utils import (
    RichHelpFormatter,
    setup_cli_logging,
    select_file_path as select_file,
    select_save_path as select_output_file,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate simulated sky temperature data based on FEKO gain patterns and GSM sky maps.",
        formatter_class=RichHelpFormatter)
    
    parser.add_argument('-i','--input', required=True, 
        help='Path to the calibrated input sky temperature file (system gain and temperature subtracted)')
    parser.add_argument('-a', '--antenna-gain', required=True,
        help='Path to the FEKO output file (CSV format) containing the gain information for the antenna')
    parser.add_argument('-o', '--output', required=True,
        help='Path to save the generated simulated sky temperature data (e.g., in pickle format)')
    
    return parser.parse_args()


def select_input_paths():
    input_path = select_file(
        title="Select the calibrated input sky temperature file",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
    )
    antenna_gain_path = select_file(
        title="Select the FEKO antenna gain CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    output_path = select_output_file(
        title="Select the output file location",
        initialfile="simulated_sky_temperature.pkl",
        defaultextension=".pkl",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
    )

    print(f"input: {input_path}")
    print(f"antenna_gain: {antenna_gain_path}")
    print(f"output: {output_path}")

    return input_path, antenna_gain_path, output_path

def load_timestamp(datapath):
    with open(datapath, "rb") as f:
        segments = pickle.load(f)
    
    utc_timestamp = segments[0]['antenna_utc_timestamps']
    antenna_T = segments[0]['ant_T_wf']
    return utc_timestamp, antenna_T

if __name__ == "__main__":
    setup_cli_logging()  # Set up logging for the script

    use_file_picker = True

    if use_file_picker:
        input_path, antenna_gain_path, output_path = select_input_paths()
    else:
        args = parse_args()  # Parse command-line arguments
        print(f"input: {args.input}")
        print(f"antenna_gain: {args.antenna_gain}")
        print(f"output: {args.output}")
