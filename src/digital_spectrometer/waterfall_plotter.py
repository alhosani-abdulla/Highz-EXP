# Adapted from Marcus's waterfall plotter for digital spectrometer data
import os
import glob
import logging
import sys
import numpy as np
from datetime import datetime, date, timedelta, timezone
import zoneinfo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from highz_exp.unit_convert import rfsoc_spec_to_dbm
from highz_exp.file_load import load_npy_dict
from file_compressor import setup_logging

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6

pjoin = os.path.join
pbase = os.path.basename

def add_timestamp(time_dir, date_str, state_no) -> dict:
    """Load all spectrum files in a time directory and return a dict with timestamp keys.

    Parameters:
    -----------
    time_dir : str
                Path to the directory containing spectrum .npy files for a specific time.

        Returns:
        --------
        loaded : dict
                Dictionary with the following structure:
                {'timestamp_str': {'spectrum': np.ndarray, ...}, 'full_timestamp': datetime, ...}
        """
    all_specs = sorted(glob.glob(pjoin(time_dir, f"*state{state_no}*")))
    loaded = {}
    for spec_file in all_specs:
        loaded.update(load_npy_dict(spec_file))
    time_dirname = pbase(time_dir)
    datestamp = datetime.strptime(date_str, '%Y%m%d').date()

    for timestamp_str in loaded.keys():
        timestamp = datetime.strptime(timestamp_str, '%H%M%S').time()
        if time_dirname.startswith("23"):
            if timestamp.hour <= 1:
                # Assign to next day
                full_timestamp = datetime.combine(
                    datestamp + timedelta(days=1), timestamp, tzinfo=zoneinfo.ZoneInfo('UTC'))
            else:
                full_timestamp = datetime.combine(
                    datestamp, timestamp, tzinfo=zoneinfo.ZoneInfo('UTC'))
        else:
            full_timestamp = datetime.combine(
                datestamp, timestamp, tzinfo=zoneinfo.ZoneInfo('UTC'))
        loaded[timestamp_str]['full_timestamp'] = full_timestamp
  
    return loaded

def get_date_state_specs(date_dir, state_indx=0):
    """Collect all spectrum files for a given date and state index."""
    all_items = glob.glob(pjoin(date_dir, "*"))
    time_dirs = [d for d in all_items if os.path.isdir(d)]
    time_dirs.sort()
    logging.info("Found %d time directories in %s", len(time_dirs), date_dir)
    if len(time_dirs) == 0:
        logging.error("No sub directories found in %s", date_dir)
        return
    
    loaded = {}
    for time_dir in time_dirs:
        loaded.update(add_timestamp(time_dir, pbase(date_dir), state_indx)) 

    return loaded

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

import plotly.graph_objects as go
import pandas as pd
import numpy as np

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


def plot_waterfall_heatmap_static(datetimes, spectra, faxis_mhz, title, output_path=None, show_plot=True, vmin=-80, vmax=-20):
    """Create a heatmap of spectra with power levels as color coding. Static version with Matplotlib without interactivity.
    
    Parameters:
    -----------
    - datetimes: np.array of datetime objects. """
    fig, ax = plt.subplots(figsize=(18, 10))

    timezone = datetimes[0].tzinfo
    time_hours = mdates.date2num(datetimes)

    # Clip spectra to maximum power level
    spectra_clipped = np.clip(spectra, vmin, vmax)

    # Format y-axis as time
    ax.yaxis_date()
    date_form = mdates.DateFormatter('%d - %H:%M', tz=timezone)
    ax.yaxis.set_major_formatter(date_form)
    ax.yaxis.set_major_locator(mdates.HourLocator(interval=2))

    # Create heatmap
    im = ax.imshow(spectra_clipped, aspect='auto', origin='lower',
                   extent=[faxis_mhz[0], faxis_mhz[-1],
                           time_hours[0], time_hours[-1]],
                   cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)

    ax.set_xlabel('Frequency (MHz)', fontsize=18)
    ax.set_ylabel(f'{timezone} Time (hours)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.set_title(title, fontsize=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dBm)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info("Heatmap saved to %s", output_path)

    if show_plot:
        plt.show()
    else:
        plt.close()

def convert_utc_list_to_local(utc_timestamps):
    """
    Converts a list of naive UTC datetime objects to local timezone-aware objects.
    """
    local_timezone = datetime.now().astimezone().tzinfo
    logging.info(f"Current timezone: {local_timezone}.")
    local_timestamps = []

    for utc_dt in utc_timestamps:
        # 1. Make the UTC datetime object timezone-aware (explicitly UTC)
        # 2. Convert to the local system's timezone
        local_aware_dt = utc_dt.astimezone(local_timezone)
        
        local_timestamps.append(local_aware_dt)
        
    return local_timestamps


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
