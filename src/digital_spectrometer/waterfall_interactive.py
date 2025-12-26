# Adapted from Marcus's waterfall plotter for digital spectrometer data
import os, logging, sys
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import argparse
from datetime import timedelta
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from highz_exp.unit_convert import convert_utc_list_to_local
from highz_exp.file_load import get_sorted_time_dirs, get_specs_from_dirs, read_loaded
from file_compressor import setup_logging
from highz_exp.spec_proc import downsample_waterfall, validate_spectra_dimensions, get_dynamic_bin_size

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6

pjoin = os.path.join
pbase = os.path.basename

def inject_gap_spacers(datetimes: List[datetime], 
    spectra: np.ndarray, threshold_seconds: float = 30.0) -> Tuple[List[datetime], np.ndarray]:
    """
    Detects temporal gaps in spectral data and injects NaN-filled rows.

    This function identifies periods where data collection was interrupted (based 
    on the threshold) and inserts a single row of NaNs at the midpoint of the gap. 
    This prevents Plotly from interpolating across the gap and visually 
    represents the interruption as a black bar.

    Args:
        datetimes: A list of datetime objects corresponding to the spectra rows.
        spectra: A 2D NumPy array of shape (N, M) containing power values.
        threshold_seconds: Gap duration in seconds required to trigger a spacer.

    Returns:
        A tuple containing:
            - A new list of datetimes (including injected midpoints).
            - A new 2D NumPy array with injected NaN rows.
    """
    new_ts: List[datetime] = []
    new_spectra_list: List[np.ndarray] = []
    num_freq_bins: int = spectra.shape[1]

    for i in range(len(datetimes)):
        # Add current real data
        new_ts.append(datetimes[i])
        new_spectra_list.append(spectra[i])

        # Check gap between current and next index
        if i < len(datetimes) - 1:
            time_diff = (datetimes[i+1] - datetimes[i]).total_seconds()
            
            if time_diff > threshold_seconds:
                # Calculate the midpoint of the interruption
                gap_midpoint = datetimes[i] + (datetimes[i+1] - datetimes[i]) / 2
                nan_row = np.full((num_freq_bins,), np.nan)
                
                new_ts.append(gap_midpoint)
                new_spectra_list.append(nan_row)

    return new_ts, np.array(new_spectra_list)


def plot_waterfall_heatmap_plotly(datetimes, spectra, faxis_mhz, title, output_path, vmin=-80, vmax=-20):
    """
    Creates an interactive waterfall plot using Plotly.
    Includes hover data, zooming, and a dynamic color-range slider.
    """

    date = datetimes[0].date()

    datetimes, spectra = inject_gap_spacers(datetimes, spectra, threshold_seconds=30)
    
    # bin_size = get_dynamic_bin_size(datetimes) 
    # datetimes, spectra = align_spectra_to_grid(datetimes, spectra, bin_size_seconds=bin_size)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(z=np.round(spectra,2), x=faxis_mhz, y=datetimes,
        colorscale='Viridis', zmin=vmin, zmax=vmax,
        colorbar=dict(title="Power (dBm)"), connectgaps=True, hoverongaps=False, zsmooth='fast',
        hovertemplate=("Time: %{y}<br>" +
            "Freq: %{x:.2f} MHz<br>" + "Power: %{z:.2f} dBm<extra></extra>"
        )
    ))
    
    # Add buttons to toggle connectgaps
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                active=0,
                font=dict(size=14, color="white", family="Arial"),
                buttons=list([
                    dict(
                        args=[{"connectgaps": True, "zsmooth": 'fast'}],
                        label="Interpolate Gaps",
                        method="restyle"
                    ),
                    dict(
                        args=[{"connectgaps": False}],
                        label="Show Gaps (Raw)",
                        method="restyle" # Use restyle to update trace attributes
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.00,
                xanchor="right",
                y=1.15,
                yanchor="top"
            ),
        ])

    # Robust logic:
    if datetimes[0] < datetimes[-1]:
        # Data is ascending (Start -> End), so reverse axis to put Start at top
        y_axis_direction = "reversed"
    else:
        # Data is already descending (End -> Start), use normal
        y_axis_direction = True

    fig.update_layout(
        title=dict(text=f"{date}: {title}", font=dict(size=24)), 
        xaxis=dict(title=dict(text="Frequency (MHz)")),
        yaxis=dict(title="Time",
            autorange=y_axis_direction,
        ),
        width=1400, height=800, template="plotly_dark",
        plot_bgcolor='black',
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

    # OR call show specifically with the renderer
    fig.write_html(output_path, auto_open=False, include_plotlyjs=True)

    logging.info("=============================================================")
    logging.info(f"Waterfall plot saved to {output_path}.")
    logging.info("=============================================================")

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

    parser.add_argument("--step_f", type=int, default=4, help="Frequency downsampling step size. Default = 4, allows 0.1 MHz resolution.")
    parser.add_argument("--step_t", type=int, default=1, help="Time downsampling step size. Default = 1.")

    # Parse the arguments
    args = parser.parse_args()

    # Initialize logic
    setup_logging()

    return args

def main(date_dir, state_indx, step_f, step_t, output_dir=None):
    # --- 1. Data Ingestion & Setup ---
    time_dirs = get_sorted_time_dirs(date_dir)
    date = pbase(date_dir)
    for quartered_time_dirs in np.array_split(time_dirs, 4):
        loaded = get_specs_from_dirs(date, quartered_time_dirs, state_indx)
        timestamps, spectra = read_loaded(loaded, sort='ascending')
        
        logging.info(f"Total spectra loaded: {len(spectra)}")
        logging.info(f"Original Timezone: {timestamps[0].tzinfo}")
        
        output_dir = output_dir or date_dir
        os.makedirs(output_dir, exist_ok=True)

        # Convert to local for processing/filenames
        local_ts = convert_utc_list_to_local(timestamps)
        logging.info(f"Local Time range: {local_ts[0]} to {local_ts[-1]}")

        # --- 2. Identify Daily Boundaries ---
        dates = np.array([dt.date() for dt in local_ts])
        change_indices = np.where(dates[:-1] != dates[1:])[0] + 1
        boundaries = [0] + list(change_indices) + [len(local_ts)]

        # --- 3. Process each Day ---
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            
            day_date = dates[start]
            day_ts = local_ts[start:end]
            day_spectra = spectra[start:end, :] # Slicing rows (time)
            
            logging.info(f"Processing Day {i}: {day_date} ({len(day_ts)} samples)")
            
            # Metadata for plotting
            f_mhz = faxis  # Assumed global or defined elsewhere
            title = f"Waterfall Plot Interactive: State {state_indx}: {day_ts[0].hour:02d} - {day_ts[-1].hour:02d})"
            output_fn = f"waterfall_{state_indx}_{day_date}_{day_ts[0].hour:02d}_{day_ts[-1].hour:02d}.html"
            
            # Validation
            if not validate_spectra_dimensions(day_ts, f_mhz, day_spectra):
                logging.warning(f"Validation failed for {day_date}: {day_ts[0].hour:02d} to {day_ts[-1].hour:02d}")
                continue

            ds_ts, ds_f, ds_spec = downsample_waterfall(day_ts, f_mhz, day_spectra, step_t=step_t, step_f=step_f)
            
            plot_waterfall_heatmap_plotly(
                ds_ts, 
                ds_spec, 
                ds_f, 
                title, 
                pjoin(output_dir, output_fn)
            )
            
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
