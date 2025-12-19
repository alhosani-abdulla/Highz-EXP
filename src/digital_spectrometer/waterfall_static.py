# Adapted from Marcus's waterfall plotter for digital spectrometer data
from matplotlib.colors import PowerNorm
import os, glob, logging, sys
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from highz_exp.unit_convert import rfsoc_spec_to_dbm, convert_utc_list_to_local
from highz_exp.file_load import get_date_state_specs
from highz_exp.plotter import plot_waterfall_heatmap_static
from file_compressor import setup_logging

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6

pjoin = os.path.join
pbase = os.path.basename

def read_loaded(loaded, date, sort='ascending') -> tuple[np.array, np.array]:
    """Read timestamps and spectra from loaded data. Sort by timestamps.

    Parameters:
    -----------
    loaded: dict. Structure {'timestamp_str': {'spectrum': np.ndarray, 'full_timestamp': datetime, ...}, ...}
    date : str. Formated like 20251216."""
    timestamps = []
    raw_timestamps_str = []
    spectra = []
    for timestamp_str, info_dict in loaded.items():
        timestamps.append(info_dict['full_timestamp'])
        spectrum = rfsoc_spec_to_dbm(info_dict['spectrum'], offset=-128)
        if len(spectrum) != nfft//2:
            logging.warning("Spectrum length %d does not match expected %d for timestamp %s", len(
                spectrum), nfft//2, timestamp_str)
            continue
        spectra.append(spectrum)

    timestamps = np.array(timestamps)
    spectra = np.array(spectra)

    sort_idx = np.argsort(
        timestamps) if sort == 'ascending' else np.argsort(timestamps)[::-1]
    if sort == 'descending':
        sort_idx = sort_idx[::-1]

    return timestamps[sort_idx], spectra[sort_idx]

def __main__(date_dir, state_indx=0, output_dir=None):
    loaded = get_date_state_specs(date_dir, state_indx=state_indx)
    timestamps, spectra = read_loaded(loaded, date=date, sort='ascending')
    logging.info("Total spectra loaded: %d", len(spectra))
    logging.info(f"The time zone is {timestamps[0].tzinfo} ")
    logging.info("Time range: %s to %s", timestamps[0], timestamps[-1])
    if output_dir is None:
        output_dir = date_dir
    else:
        if not os.path.isdir(output_dir):
            os.mkdirs(output_dir)
   
    plot_waterfall_heatmap_static(timestamps, spectra, faxis,
                        title=f'Waterfall Plot {os.path.basename(date_dir)}: State {state_indx}',
                        output_path=pjoin(output_dir, f'waterfall_state{state_indx}_GMT.png'),
                        show_plot=False) 
        
    # Convert to local timezone
    local_timestamps = convert_utc_list_to_local(timestamps)
    logging.info("Time range: %s to %s", local_timestamps[0], local_timestamps[-1])
    plot_waterfall_heatmap_static(local_timestamps, spectra, faxis,
    					   title=f'Waterfall Plot {os.path.basename(date_dir)}: State {state_indx}',
    					   output_path=pjoin(output_dir, f'waterfall_state{state_indx}_localtime.png'),
    					   show_plot=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python waterfall_plotter.py <date_directory> [state_index]")
        sys.exit(1)

    setup_logging()
    input_dir = sys.argv[1]
    state_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None 

    __main__(input_dir, state_indx=state_index, output_dir=output_dir)
