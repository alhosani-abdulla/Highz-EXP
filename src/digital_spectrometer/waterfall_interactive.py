# Adapted from Marcus's waterfall plotter for digital spectrometer data
import os, logging, sys
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import argparse, statistics
from datetime import timedelta
from highz_exp.unit_convert import rfsoc_spec_to_dbm, convert_utc_list_to_local
from highz_exp.file_load import get_date_state_specs
from file_compressor import setup_logging
from highz_exp.spec_proc import downsample_waterfall, validate_spectra_dimensions

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6

pjoin = os.path.join
pbase = os.path.basename

def read_loaded(loaded, sort='ascending') -> tuple[np.array, np.array]:
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

def align_spectra_to_grid(datetimes, spectra, bin_size_seconds=8):
    """
    Creates a regular grid of timestamps and fills missing gaps with NaN.
    """
    if not datetimes:
        return [], []

    # 1. Create a regular time grid from start to end
    start_time = datetimes[0]
    end_time = datetimes[-1]
    
    # Calculate total expected steps
    total_seconds = int((end_time - start_time).total_seconds())
    num_steps = (total_seconds // bin_size_seconds) + 1
    
    # Generate the regular grid
    regular_datetimes = [start_time + timedelta(seconds=i * bin_size_seconds) 
                         for i in range(num_steps)]
    
    # 2. Prepare a new matrix filled with NaN
    # Shape: (number of time steps, number of frequency bins)
    num_freq_bins = spectra.shape[1]
    aligned_spectra = np.full((num_steps, num_freq_bins), np.nan)

    # 3. Map original spectra to the nearest grid index
    for i, dt in enumerate(datetimes):
        # Calculate which grid index this actual timestamp belongs to
        delta_seconds = (dt - start_time).total_seconds()
        grid_idx = int(round(delta_seconds / bin_size_seconds))
        
        # Ensure we don't go out of bounds due to rounding
        if grid_idx < num_steps:
            aligned_spectra[grid_idx, :] = spectra[i, :]

    return regular_datetimes, aligned_spectra

def plot_waterfall_heatmap_plotly(datetimes, spectra, faxis_mhz, title, output_path, vmin=-80, vmax=-20):
    """
    Creates an interactive waterfall plot using Plotly.
    Includes hover data, zooming, and a dynamic color-range slider.
    """

    date = datetimes[0].date()
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(z=np.round(spectra,2), x=faxis_mhz, y=datetimes,
        colorscale='Viridis', zmin=vmin, zmax=vmax,
        colorbar=dict(title="Power (dBm)"),
        hovertemplate=(
            "Time: %{y}<br>" +
            "Freq: %{x:.2f} MHz<br>" +
            "Power: %{z:.2f} dBm<extra></extra>"
        )
    ))

    # Robust logic:
    if datetimes[0] < datetimes[-1]:
        # Data is ascending (Start -> End), so reverse axis to put Start at top
        y_axis_direction = "reversed"
    else:
        # Data is already descending (End -> Start), use normal
        y_axis_direction = True

    fig.update_layout(
        title=dict(text=f"{date}: {title}", font=dict(size=24)), 
        xaxis=dict(title=dict(text="Frequency (MHz)")
                   ),
        yaxis=dict(
            title="Time",
            autorange=y_axis_direction,
        ),
        width=1400, height=800, template="plotly_dark",
        margin=dict(t=150, b=150)
    )
    
    # Update X-axis label font size
    fig.update_xaxes(title_font={"size": 20})

    # Update Y-axis label font size
    fig.update_yaxes(title_font={"size": 20})

    fig.update_xaxes(tickfont=dict(size=16))
    fig.update_yaxes(tickfont=dict(size=16))

    # 1. Define steps for zmin (Floor)
    min_steps = []
    for val in range(-90, -30, 5):
        min_steps.append({
            "method": "restyle",
            "label": str(val),
            "args": [{"zmin": val}]
        })

    # 2. Define steps for zmax (Ceiling)
    max_steps = []
    for val in range(-80, -10, 5):
        max_steps.append({
            "method": "restyle",
            "label": str(val),
            "args": [{"zmax": val}]
        })

    # 3. Add both to the layout
    fig.update_layout(
        margin=dict(b=150),
        sliders=[
            # Slider for zmin
            {
                "active": 0,
                "currentvalue": {"prefix": "Min Power (Floor): "},
                "pad": {"t": 50},
                "len": 0.45,
                "x": 0,
                "steps": min_steps
            },
            # Slider for zmax
            {
                "active": 5,
                "currentvalue": {"prefix": "Max Power (Ceiling): "},
                "pad": {"t": 50},
                "len": 0.45,
                "x": 0.55,
                "steps": max_steps
            }
        ]
    )

    # # Gradient Adjustment Buttons
    # fig.update_layout(margin=dict(t=100, b=100),
    #     updatemenus=[
    #         dict(
    #             type="buttons",
    #             direction="left",
    #             active=0, x=0.5, y=-0.5,
    #             xanchor="center",
    #             yanchor="bottom",
    #             buttons=[
    #                 # Default Range
    #                 dict(label="Reset Range (Min to Max)",
    #                      method="restyle",
    #                      args=[{"zmin": vmin, "zmax": vmax}]),
                    
    #                 # Narrow Range (High Contrast - useful for faint signals)
    #                 dict(label="High Contrast [-50, -30]",
    #                      method="restyle",
    #                      args=[{"zmin": -50, "zmax": -30}]),
                    
    #                 # Wide Range (Deep Noise Floor)
    #                 dict(label="Wide Range [-80, -20]",
    #                      method="restyle",
    #                      args=[{"zmin": -80, "zmax": -20}]),
                    
    #                 # See Noise Floor (Lower floor)
    #                 dict(label="Low Power Level [-80, -60]",
    #                      method="restyle",
    #                      args=[{"zmin": -80, "zmax": -60}]),
                    
    #                 # See Noise Floor (Lower floor)
    #                 dict(label="Absolute Noise Floor [-90, -70]",
    #                      method="restyle",
    #                      args=[{"zmin": -90, "zmax": -70}])
    #             ]
    #         )
    #     ]
    # )

    # OR call show specifically with the renderer
    fig.write_html(output_path, auto_open=True, include_plotlyjs='cdn')

def main_cli():
    parser = argparse.ArgumentParser(
        description="Waterfall Plotter for Digital Spectrometer Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

    parser.add_argument("--step_f", type=int, default=4, help="Frequency downsampling step size")

    parser.add_argument(
        "--step_t",
        type=int,
        default=5,
        help="Time downsampling step size"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Initialize logic
    setup_logging()

    return args

def main(date_dir, state_indx, step_f, step_t, output_dir=None):
    loaded = get_date_state_specs(date_dir, state_indx=state_indx)
    timestamps, spectra = read_loaded(loaded, sort='ascending')
    logging.info("Total spectra loaded: %d", len(spectra))
    logging.info(f"The time zone is {timestamps[0].tzinfo} ")
    logging.info("Time range: %s to %s", timestamps[0], timestamps[-1])
    if output_dir is None:
        output_dir = date_dir
    else:
        if not os.path.isdir(output_dir):
            os.mkdirs(output_dir)
        
    # Convert to local timezone
    local_timestamps = convert_utc_list_to_local(timestamps)
    logging.info("Time range: %s to %s", local_timestamps[0], local_timestamps[-1])

    date = pbase(date_dir)
    output_path = pjoin(output_dir, f'Waterfall_{date}_state{state_indx}.html')

    # 1. Convert datetimes to just dates (in a numpy-friendly format)
    dates = np.array([dt.date() for dt in local_timestamps])

    # 2. Find where the date changes (the "break points")
    # np.where(dates[:-1] != dates[1:])[0] finds the index BEFORE the change
    change_indices = np.where(dates[:-1] != dates[1:])[0] + 1

    # 3. Add the start and end indices to create boundaries
    boundaries = [0] + list(change_indices) + [len(local_timestamps)]

    # 4. Iterate over the boundaries to slice the data
    for i in range(len(boundaries) - 1):
        f_mhz = faxis
        title = f"Waterfall Plot Interactive: state {state_indx}"
        start, end = boundaries[i], boundaries[i+1]
        
        current_date = dates[start]
        daily_ts = local_timestamps[start:end]
        logging.info(f"{i} ---- Time range for current slice: {daily_ts[0]} to {daily_ts[-1]}")
        daily_spectra = spectra[start:end, :]  # Fast contiguous memory slice
        
        # Run your plotting function
        output_fn = f"waterfall_{state_indx}_{current_date}_{daily_ts[-1].hour}.html"
        if_valid = validate_spectra_dimensions(daily_ts, f_mhz, daily_spectra)
        daily_ts, f_mhz, daily_spectra = downsample_waterfall(daily_ts, f_mhz, daily_spectra, step_f=step_f, step_t=step_t)
        
        plot_waterfall_heatmap_plotly(daily_ts, daily_spectra, f_mhz, title, pjoin(output_dir, output_fn))

    # if_valid = validate_spectra_dimensions(local_timestamps, faxis_mhz=f_mhz, spectra=spectra)
    # local_timestamps, f_mhz, spectra = downsample_waterfall(local_timestamps, f_mhz, spectra, step_f=step_f, step_t=step_t)
    
    # plot_waterfall_heatmap_plotly(local_timestamps, spectra, f_mhz, f"Waterfall Plot Interactive: state {state_indx}", output_path=output_path)

if __name__ == "__main__":

    args = main_cli()

    # Call main with the parsed values
    # Note: I've updated the parameter names to match your main function's signature
    main(
        date_dir=args.input_dir, 
        state_indx=args.state_index, 
        output_dir=args.output_dir,
        step_f=args.step_f,
        step_t=args.step_t
    )
