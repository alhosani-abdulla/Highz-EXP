# Adapted from Marcus's waterfall plotter for digital spectrometer data
import os, logging, sys
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import argparse
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

def plot_waterfall_heatmap_plotly(datetimes, spectra, faxis_mhz, title, output_path, vmin=-80, vmax=-20):
    """
    Creates an interactive waterfall plot using Plotly.
    Includes hover data, zooming, and a dynamic color-range slider.
    """

    date = datetimes[0].date()
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(z=spectra, x=faxis_mhz, y=datetimes,
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
        margin=dict(t=50, b=50)
    )
    
    # Update X-axis label font size
    fig.update_xaxes(title_font={"size": 20})

    # Update Y-axis label font size
    fig.update_yaxes(title_font={"size": 20})

    fig.update_xaxes(tickfont=dict(size=16))
    fig.update_yaxes(tickfont=dict(size=16))
    
    # Gradient Adjustment Buttons
    fig.update_layout(margin=dict(t=10, b=10),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                active=0, x=0.5, y=-0.5,
                xanchor="center",
                yanchor="bottom",
                buttons=[
                    # Default Range
                    dict(label="Reset Range (Min to Max)",
                         method="restyle",
                         args=[{"zmin": vmin, "zmax": vmax}]),
                    
                    # Narrow Range (High Contrast - useful for faint signals)
                    dict(label="High Contrast [-50, -30]",
                         method="restyle",
                         args=[{"zmin": -50, "zmax": -30}]),
                    
                    # Wide Range (Deep Noise Floor)
                    dict(label="Wide Range [-80, -20]",
                         method="restyle",
                         args=[{"zmin": -80, "zmax": -20}]),
                    
                    # See Noise Floor (Lower floor)
                    dict(label="Noise Floor [-80, -60]",
                         method="restyle",
                         args=[{"zmin": -80, "zmax": -60}])
                ]
            )
        ]
    )

    # OR call show specifically with the renderer
    fig.write_html(output_path, auto_open=True)

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
    output_path = pjoin(output_dir, f'Waterfall_{date}.html')

    f_mhz = faxis

    if_valid = validate_spectra_dimensions(local_timestamps, faxis_mhz=f_mhz, spectra=spectra)
    local_timestamps, f_mhz, spectra = downsample_waterfall(local_timestamps, f_mhz, spectra, step_f=step_f, step_t=step_t)
    plot_waterfall_heatmap_plotly(local_timestamps, spectra, f_mhz, "Waterfall Plot Interactive", output_path=output_path)

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
