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

def plot_waterfall_heatmap_plotly(datetimes, spectra, faxis_mhz, title, vmin=-80, vmax=-20):
    """
    Creates an interactive waterfall plot using Plotly.
    Includes hover data, zooming, and a dynamic color-range slider.
    """
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=spectra,
        x=faxis_mhz,
        y=datetimes,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(title="Power (dBm)"),
        hovertemplate=(
            "Time: %{y}<br>" +
            "Freq: %{x:.2f} MHz<br>" +
            "Power: %{z:.2f} dBm<extra></extra>"
        )
    ))

    # Logic to determine if we need to reverse the time axis
    # If the first timestamp is later than the last, we reverse to keep time flowing 'up'
    is_reversed = "reversed" if datetimes[0] > datetimes[-1] else True

    fig.update_layout(
        title=title, xaxis=dict(title="Frequency (MHz)"),
        yaxis=dict(
            title="Time",
            autorange=is_reversed  # Use True instead of "normal"
        ),
        width=1000, height=700, template="plotly_dark"
    )


    # Enable the color-range adjustment (UI buttons/sliders)
    # Note: Plotly's 'edit' mode allows users to click the colorbar 
    # and drag the limits, but we can also add custom buttons:
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"zmin": vmin, "zmax": vmax}],
                        label="Reset Range",
                        method="restyle"
                    ),
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    fig.show()



def __main__(date_dir, state_indx=0):
    loaded = get_date_state_specs(date_dir, state_indx=state_indx)
    timestamps, spectra = read_loaded(loaded, date=date, sort='ascending')
    logging.info("Total spectra loaded: %d", len(spectra))
    logging.info("Time range: %s to %s", timestamps[0], timestamps[-1])
    # plot_waterfall_heatmap_static(timestamps, spectra, faxis,
    # 					   title=f'Waterfall Plot {os.path.basename(date_dir)}: State {state_indx}',
    # 					   output_path=pjoin(date_dir, f'waterfall_state{state_indx}.png'),
    # 					   show_plot=False)
    # Convert to local timezone and try again
    local_timestamps = convert_utc_list_to_local(timestamps)
    logging.info("Time range: %s to %s", local_timestamps[0], local_timestamps[-1])
    # plot_waterfall_heatmap_static(local_timestamps, spectra, faxis,
    # 					   title=f'Waterfall Plot {os.path.basename(date_dir)}: State {state_indx}',
    # 					   output_path=pjoin(date_dir, f'waterfall_state{state_indx}_localtime.png'),
    # 					   show_plot=False)
    plot_waterfall_heatmap_plotly(local_timestamps, spectra, faxis, "Interactive RF Spectrum")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python waterfall_plotter.py <date_directory> [state_index]")
        sys.exit(1)

    setup_logging()
    input_dir = sys.argv[1]
    state_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    __main__(input_dir, state_indx=state_index)
