# Adapted from Marcus's waterfall plotter for digital spectrometer data
import os, logging
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Tuple

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

def plot_waterfall_heatmap_plotly(
    datetimes: List[datetime],
    spectra: np.ndarray,
    faxis_mhz: np.ndarray,
    title: str, 
    output_path: str,
    unit="dBm",
    vmin: int = -80,
    vmax: int = -20,
    step: int = 5,
) -> None:
    """
    Creates an interactive waterfall plot using Plotly.
    
    Generates a heatmap visualization of spectral data over time with interactive
    features including hover data, zooming, gap visualization, and dynamic color-range sliders.

    Args:
        datetimes: List of datetime objects corresponding to each spectrum row.
        spectra: 2D NumPy array of shape (N, M) containing power values in dBm.
        faxis_mhz: 1D NumPy array of frequency values in MHz for the x-axis.
        title: Title text for the plot.
        output_path: File path where the HTML plot will be saved.
        unit: Unit for the power values. Defaults to "dBm".
        vmin: Minimum power value (dBm) for the color scale. Defaults to -80.
        vmax: Maximum power value (dBm) for the color scale. Defaults to -20.

    Returns:
        None. Writes the plot to an HTML file.
    """
    date = datetimes[0].date()
    
    cbar_title = f"Power ({unit})"

    datetimes, spectra = inject_gap_spacers(datetimes, spectra, threshold_seconds=30)
    
    # bin_size = get_dynamic_bin_size(datetimes) 
    # datetimes, spectra = align_spectra_to_grid(datetimes, spectra, bin_size_seconds=bin_size)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(z=np.round(spectra, 2), x=faxis_mhz, y=datetimes,
        colorscale='Viridis', zmin=vmin, zmax=vmax,
        colorbar=dict(title=cbar_title), connectgaps=True, hoverongaps=False, zsmooth='fast',
        hovertemplate=("Time: %{y}<br>" +
            "Freq: %{x:.2f} MHz<br>" + f"Power: %{{z:.2f}} {unit}<extra></extra>"
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
                        method="restyle"
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

    # Robust logic: reverse y-axis if data is ascending to put start at top
    y_axis_direction = "reversed" if datetimes[0] < datetimes[-1] else True

    fig.update_layout(
        title=dict(text=f"{date}: {title}", font=dict(size=24)), 
        xaxis=dict(title=dict(text="Frequency (MHz)")),
        yaxis=dict(title="Time", autorange=y_axis_direction),
        width=1400, height=800, template="plotly_dark",
        plot_bgcolor='black',
        margin=dict(t=150, b=150)
    )
    
    # Update axis label and tick font sizes
    fig.update_xaxes(title_font={"size": 20}, tickfont=dict(size=16))
    fig.update_yaxes(title_font={"size": 20}, tickfont=dict(size=16))

    # Define slider steps for zmin (Floor)
    min_steps = [
        {"method": "restyle", "label": str(val), "args": [{"zmin": val}]}
        for val in range(vmin - 10, vmax - 10, step)
    ]

    # Define slider steps for zmax (Ceiling)
    max_steps = [
        {"method": "restyle", "label": str(val), "args": [{"zmax": val}]}
        for val in range(vmin, vmax, step)
    ]

    # Add sliders to layout
    fig.update_layout(
        margin=dict(b=150),
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Min Power (Floor): "},
                "pad": {"t": 50},
                "len": 0.45,
                "x": 0,
                "steps": min_steps
            },
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

    fig.write_html(output_path, auto_open=False, include_plotlyjs=True)

    logging.info("=============================================================")
    logging.info(f"Waterfall plot saved to {output_path}.")
    logging.info("=============================================================")
