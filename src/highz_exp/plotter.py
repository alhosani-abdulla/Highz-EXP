import matplotlib.pyplot as plt
import numpy as np
import os, re
import pygdsm
import healpy as hp
from pathlib import Path
from typing import Union
from matplotlib.colors import PowerNorm
from matplotlib.ticker import NullLocator
import matplotlib.dates as mdates
from datetime import datetime
from typing import List
import logging
import skrf as rf

from .spec_class import Spectrum

from os.path import join as pjoin, basename as pbase

LEGEND = ['6" shorted', "8' cable open",'Black body','Ambient temperature load','Noise diode',"8' cable short",'Open Circuit state']

DEFAULT_MATPLOTLIB_STYLE = {
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 20,
}

def set_matplotlib_defaults(style_overrides: dict | None = None) -> dict:
    """Apply and return global matplotlib defaults for plotting in this module."""
    style = DEFAULT_MATPLOTLIB_STYLE.copy()
    if style_overrides is not None:
        style.update(style_overrides)
    plt.rcParams.update(style)
    return style

set_matplotlib_defaults()

def plot_measured_vs_fitted(ntwk_dict, scale='linear', save_plot=True, save_path=None, ylabel='Magnitude', title='Measured vs Fitted Spectrum', show_residual=False, show_bottom_panel=True, return_handles=False):
    """
    Plot magnitude for measured and fitted spectrum data, and optionally a ratio panel (measured/fitted) or residual panel.

    Parameters:
    - ntwk_dict (dict): {'measured': skrf.Network, 'fitted': skrf.Network}
    - show_residual (bool): If True, show residual (measured - fitted). If False, show ratio (fitted/measured).
    - show_bottom_panel (bool): Whether to show the bottom panel (ratio or residual).
    """
    assert len(ntwk_dict) == 2, "ntwk_dict must contain exactly two items: measured and fitted."
    keys = list(ntwk_dict.keys())
    measured_ntwk = ntwk_dict[keys[0]]
    fitted_ntwk = ntwk_dict[keys[1]]

    freq = measured_ntwk.f
    spec_measured = measured_ntwk.s[:, 0, 0]
    spec_fitted = fitted_ntwk.s[:, 0, 0]

    mag_measured = 20 * np.log10(np.abs(spec_measured)) if scale == 'log' else np.abs(spec_measured)
    mag_fitted = 20 * np.log10(np.abs(spec_fitted)) if scale == 'log' else np.abs(spec_fitted)

    nrows = 2 if show_bottom_panel else 1
    fig, axes = plt.subplots(nrows=nrows, figsize=(10, 7), sharex=True)
    if nrows == 1:
        axes = [axes]  # Make it iterable for consistency

    ax_mag = axes[0]

    ax_mag.plot(freq / 1e6, mag_measured, label=f'{keys[0]}', color='C0')
    ax_mag.plot(freq / 1e6, mag_fitted, label=f'{keys[1]}', color='C1', linestyle='--')
    ax_mag.set_ylabel(ylabel)
    ax_mag.legend(loc='best')
    ax_mag.grid(True)

    if show_bottom_panel:
        ax_bottom = axes[1]
        
        if show_residual:
            residual = mag_measured - mag_fitted
            ax_bottom.plot(freq / 1e6, residual, color='C2')
            ax_bottom.axhline(0, color='red', linestyle='-', linewidth=1.5, label='residual = 0')
            ax_bottom.set_ylabel('Residual (Measured - Theory)')
        else:
            ratio = mag_measured / mag_fitted
            ax_bottom.plot(freq / 1e6, 1 / ratio, color='C2')
            ax_bottom.axhline(1, color='red', linestyle='-', linewidth=1.5, label='measured/theory = 1')
            ax_bottom.set_ylabel('Measured/Theory')
        
        ax_bottom.set_xlabel('Frequency [MHz]')
        ax_bottom.grid(True)
        ax_bottom.legend(loc='best')
    else:
        ax_mag.set_xlabel('Frequency [MHz]')

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plot:
        if save_path is not None:
            plt.savefig(save_path)
        else:
            print("! Save path not entered.")

    plt.show()

    if return_handles:
        return fig, axes

def plot_network_data(ntwk_dict, save_plot=True, show_phase=True, save_path=None, ylabel='Magnitude', title='Network Data', s_param=(0, 0),
                      ylim=None, return_handles=False):
    """
    Plot magnitude (and optionally phase) from a dictionary of scikit-rf Network objects.
    Can be used for S11, gain, power spectrum, or any S-parameter data.

    Parameters:
    - ntwk_dict (dict): {label: skrf.Network}. Frequency points are in Hz.
    - save_plot (bool): Whether to save the plot.
    - show_phase (bool): Whether to show phase subplot.
    - save_path (str): Path to save the plot.
    - ylabel (str): Y-axis label for magnitude plot.
    - title (str): Plot title.
    - s_param (tuple): S-parameter indices (i, j) to plot. Default (0, 0) for S11.
    """

    nrows = 2 if show_phase else 1
    fig, axes = plt.subplots(nrows=nrows, figsize=(12, 8), sharex=True)
    if nrows == 1:
        axes = [axes]  # Make it iterable for consistency

    ax_mag = axes[0]
    ax_phase = axes[1] if show_phase else None

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (label, ntwk) in enumerate(ntwk_dict.items()):
        freq = ntwk.f  # in Hz
        s_data = ntwk.s[:, s_param[0], s_param[1]]

        magnitude = np.abs(s_data)
        phase = np.angle(s_data, deg=True)

        color = color_cycle[idx % len(color_cycle)]

        ax_mag.plot(freq / 1e6, magnitude, label=f'{label}', color=color)
        if show_phase:
            ax_phase.plot(freq / 1e6, phase, label=f'{label}', color=color, linestyle='--')

    ax_mag.set_ylabel(ylabel)
    ax_mag.grid(True)
    ax_mag.legend(loc='best')

    if ylim is not None:
        ax_mag.set_ylim(ylim)

    if show_phase:
        ax_phase.set_xlabel('Frequency [MHz]')
        ax_phase.set_ylabel('Phase [deg]')
        ax_phase.grid(True)
        ax_phase.legend(loc='best')

    ax_mag.set_xlabel('Frequency [MHz]')

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plot:
        if save_path is not None:
            plt.savefig(save_path)
        else:
            print("! Save path not entered.")

    plt.show()

    if return_handles:
        return fig, axes

def plot_load_s2p(file_path, db=True, x_scale='linear', title='Gain Measurement (S21)', ymax=None, ymin=None, show_phase=False, attenuation=0, save_dir=None, suffix=None, return_handles=False) -> rf.Network:
    """
    Plot and load gain from a .s2p file (or list of .s2p files) using scikit-rf.

    Parameters:
    - file_path (str/list): Path to the .s2p file
    - db (bool): If True, plot gain in dB
    - show_phase (bool): If True, also plot phase in degrees
    - attenuation (float): Attenuation that was applied to the gain measurements

    Returns:
    - rf.Network: Loaded network object, with S21 representing the gain
    """
    # Load 2-port network
    if isinstance(file_path, str):
        network = rf.Network(file_path)
        freq = network.f
        s21 = network.s[:, 1, 0]  # S21 = port 2 output / port 1 input
        mag = 20 * np.log10(np.abs(s21)) + attenuation if db else np.abs(s21)
        phase = np.angle(s21, deg=True)
        parent_dir = os.path.dirname(file_path)
    elif isinstance(file_path, list):
        networks = [None]
        networks[0] = rf.Network(file_path[0])
        for file in file_path[1:]:
            network = rf.Network(file)
            network.interpolate(networks[0].f)
            networks.append(network)
        s21 = networks[0].s[:, 1, 0]
        for network in networks[1:]:
            s21 *= network.s[:, 1, 0]
        mag = 20 * np.log10(np.abs(s21)) + attenuation if db else np.abs(s21)
        freq = networks[0].f
        phase = np.angle(s21, deg=True)
        parent_dir = os.path.dirname(file_path[0])

    fig, ax1 = plt.subplots()
    fig.set_size_inches(14, 8)
    if x_scale == 'log':
        ax1.set_xscale('log')
    ax1.plot(freq / 1e6, mag, label='Gain (S21)' + (' [dB]' if db else ''))
    ax1.set_xlabel('Frequency [MHz]')
    ax1.set_ylabel('Gain' + (' [dB]' if db else ''))
    if ymax is not None:
        ax1.set_ylim(top=ymax)
    if ymin is not None:
        ax1.set_ylim(bottom=ymin)
    ax1.grid(True)

    marker_freqs_mhz = [20, 200]
    for f_mhz in marker_freqs_mhz:
      # Find closest index
      target_freq_hz = f_mhz * 1e6
      idx = np.argmin(np.abs(freq - target_freq_hz))
      marker_gain = mag[idx]
      marker_freq_ghz = freq[idx] / 1e6

      # Plot marker
      ax1.plot(marker_freq_ghz, marker_gain, 'ro')
      ax1.annotate(f'{marker_gain:.2f} dB\n@ {f_mhz:.0f} MHz',
                    (marker_freq_ghz, marker_gain),
                    textcoords="offset points", xytext=(10, 10), ha='left',
                    fontsize=16, color='darkred')
    ax2 = None
    if show_phase:
        ax2 = ax1.twinx()
        ax2.plot(freq / 1e9, phase, color='r', linestyle='--', label='Phase [deg]')
        ax2.set_ylabel('Phase [deg]', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

    plt.title(title)
    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(f'{save_dir}/Gain_{suffix}.png')
    plt.show()

    if return_handles:
        return network, fig, (ax1, ax2)
    return network

def plot_spectra(loaded_specs:list[Spectrum], save_path=None, ylabel=None, y_range=None,
        marker_freqs=None, freq_range=(None, None), yticks=None, 
        title='Recorded Spectrum', show_plot=True, return_handles=False,
        **kwargs):
    """Plot the spectrum from a dictionary of scikit-rf Network objects and save the figure if save_dir is not None.
    
    Parameters:
        - loaded_specs: list of Spectrum objects to plot, with frequency in Hz.
        - save_path (str, optional): Path to save the plot. If None, the plot is not saved.
        - ylabel (str, optional): Y-axis label. If None, defaults to 'PSD [dBm]'.
        - y_range (tuple, optional): Y-axis range to plot (ymin, ymax).
        - freq_range (tuple, optional): Frequency range to plot (fmin, fmax) in MHz
        - marker_freqs (list, optional): Frequencies in MHz to place markers on the plot.
        - yticks (list, optional): Y-axis ticks to set.
        - show_plot (bool): Whether to display the plot. If False, the plot is closed after saving.
        - kwargs (dict): Additional keyword arguments for plt.plot() when plotting the spectra.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, spec in enumerate(loaded_specs):
        freq = spec.freq  # in Hz
        spectrum = spec.spec
        
        # Convert frequency to MHz for plotting
        faxis_mhz = freq / 1e6
        
        if spec.colorcode is not None:
            color = spec.colorcode
        else:
            color = color_cycle[idx % len(color_cycle)]
        
        if freq_range[0] is not None:
            valid_idx = (faxis_mhz >= freq_range[0])
            faxis_mhz = faxis_mhz[valid_idx]
            spectrum = spectrum[valid_idx]
            
        ax.plot(faxis_mhz, spectrum, label=spec.name, color=color, **kwargs)
        
        ymin, ymax = y_range if y_range is not None else (None, None)
        
        # Dynamically adjust y-axis limits based on the data if not provided
        if ymax is None:
            ymax_state = np.max(spectrum)
            if ymax_state > (ymax or -np.inf): 
                ymax = ymax_state

        if ymin is None:
            ymin_state = np.min(spectrum)
            if ymin_state < (ymin or np.inf): 
                ymin = ymin_state
        
        # Plot markers if specified
        if marker_freqs is not None:
            for mf in marker_freqs:
                # Find closest index
                target_freq_hz = mf * 1e6
                idx = np.argmin(np.abs(freq - target_freq_hz))
                marker_psd = spectrum[idx]
                marker_freq_mhz = freq[idx] / 1e6

                # Plot marker
                ax.plot(marker_freq_mhz, marker_psd, 'ro')
                ax.annotate(f'{marker_psd:.2f} \n@ {mf:.0f} MHz',
                            (marker_freq_mhz, marker_psd),
                            textcoords="offset points", xytext=(10, 10), ha='left',
                            fontsize=16, color='darkred')

    ylim = (ymin, ymax)
    if ylabel is None:
        ylabel = 'PSD [dBm]'

    ax.set_ylim(*ylim)
    if freq_range is not None:
        ax.set_xlim(right=freq_range[1])
    ax.legend(ncol=2, loc='best')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Frequency [MHz]')
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    
    # Save the plot
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()

    if return_handles:
        return fig, ax

def plot_two_spectra_with_residual(spec_a: Spectrum, spec_b: Spectrum, save_path=None,
        ylabel='PSD [dBm]', residual_ylabel='Residual [dB]', y_range=None,
        residual_y_range=None, freq_range=(None, None), yticks=None,
        marker_freqs=None, title='Recorded Spectra Comparison', show_plot=True,
        residual_label=None, return_handles=False, **kwargs):
    """Plot two spectra and a residual panel defined as spec_a - spec_b.

    Parameters:
        - spec_a (Spectrum): First spectrum object.
        - spec_b (Spectrum): Second spectrum object.
        - save_path (str, optional): Path to save the plot. If None, the plot is not saved.
        - y_range (tuple, optional): Y-axis range for the top panel (ymin, ymax).
        - residual_y_range (tuple, optional): Y-axis range for residual panel (ymin, ymax).
        - freq_range (tuple, optional): Frequency range to plot (fmin, fmax) in MHz.
        - marker_freqs (list, optional): Frequencies in MHz to place residual markers.
        - show_plot (bool): Whether to display the plot. If False, the plot is closed after saving.
        - kwargs (dict): Additional keyword arguments for top-panel line plots.
    """
    freq_a_hz = spec_a.freq
    spec_a_vals = spec_a.spec

    freq_b_hz = spec_b.freq
    spec_b_vals = spec_b.spec

    # Interpolate spec_b onto spec_a frequency grid if needed.
    if len(freq_a_hz) != len(freq_b_hz) or not np.allclose(freq_a_hz, freq_b_hz):
        spec_b_vals = np.interp(freq_a_hz, freq_b_hz, spec_b_vals)

    faxis_mhz = freq_a_hz / 1e6
    residual = spec_a_vals - spec_b_vals

    start_idx, end_idx = _get_freq_range_indices(faxis_mhz, freq_range)
    faxis_mhz = faxis_mhz[start_idx:end_idx + 1]
    spec_a_vals = spec_a_vals[start_idx:end_idx + 1]
    spec_b_vals = spec_b_vals[start_idx:end_idx + 1]
    residual = residual[start_idx:end_idx + 1]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                             gridspec_kw={'height_ratios': [3, 1]})

    color_a = spec_a.colorcode if spec_a.colorcode is not None else 'C0'
    color_b = spec_b.colorcode if spec_b.colorcode is not None else 'C1'

    ax_top.plot(faxis_mhz, spec_a_vals, label=spec_a.name, color=color_a, **kwargs)
    ax_top.plot(faxis_mhz, spec_b_vals, label=spec_b.name, color=color_b, **kwargs)
    ax_top.set_ylabel(ylabel)
    ax_top.grid(True)
    ax_top.legend(loc='best')

    if y_range is not None:
        ax_top.set_ylim(*y_range)
    if yticks is not None:
        ax_top.set_yticks(yticks)

    if residual_label is None:
        residual_label = f'{spec_a.name} - {spec_b.name}'

    ax_bottom.plot(faxis_mhz, residual, color='C2', label=residual_label)
    ax_bottom.axhline(0, color='red', linestyle='-', linewidth=1.2)
    ax_bottom.set_ylabel(residual_ylabel)
    ax_bottom.set_xlabel('Frequency [MHz]')
    ax_bottom.grid(True)
    ax_bottom.legend(loc='best')

    if residual_y_range is not None:
        ax_bottom.set_ylim(*residual_y_range)

    if marker_freqs is not None:
        for mf in marker_freqs:
            idx = np.argmin(np.abs(faxis_mhz - mf))
            marker_residual = residual[idx]
            marker_freq_mhz = faxis_mhz[idx]
            ax_bottom.plot(marker_freq_mhz, marker_residual, 'ro', markersize=7, zorder=10)
            ax_bottom.annotate(f'{marker_residual:.2f} @ {mf:.0f} MHz',
                               (marker_freq_mhz, marker_residual),
                               textcoords='offset points', xytext=(10, 10), ha='left',
                               fontsize=14, color='darkred')

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    if return_handles:
        return fig, (ax_top, ax_bottom)

def _get_freq_range_indices(f_mhz: np.ndarray, freq_range: tuple = (None, None)) -> tuple[int, int]:
    """Return inclusive start/end indices closest to the requested frequency range."""
    if freq_range is None:
        return 0, len(f_mhz) - 1

    start_freq, end_freq = freq_range

    if start_freq is None:
        start_idx = 0
    else:
        start_idx = np.argmin(np.abs(f_mhz - start_freq))

    if end_freq is None:
        end_idx = len(f_mhz) - 1
    else:
        end_idx = np.argmin(np.abs(f_mhz - end_freq))

    return start_idx, end_idx

def plot_gain(f_mhz, gain, label=None, freq_range=(None, None), y_range=(None, None),
              xlabel='Frequency (MHz)', ylabel='Gain (dB)', title=None, save_path=None, 
              marker_freqs=None, return_handles=False, **plot_kwargs):
    """Plot gain over a specified frequency range.
    
    Parameters:
        - f_mhz (np.ndarray): Frequency axis in MHz.
        - freq_range (tuple, optional): Frequency range to plot (fmin, fmax) in MHz. If (None, None), plots the entire range of f_mhz.
        - gain (np.ndarray or list of np.ndarray): Gain values.
        - marker_freqs (list, optional): Frequencies in MHz to place markers on the plot.
    """
    start_idx, end_idx = _get_freq_range_indices(f_mhz, freq_range)
    fig, ax = plt.subplots(figsize=(12, 8))
    is_multi = isinstance(gain, list)
    if not is_multi:
        ax.errorbar(f_mhz[start_idx:end_idx+1], gain[start_idx:end_idx+1], **plot_kwargs)
    else:
        if label is None:
            labels = [f'gain_{i}' for i in range(len(gain))]
        else:
            labels = list(label)
            if len(labels) != len(gain):
                raise ValueError("label length must match number of gain arrays")
        for g, lab in zip(gain, labels):
            ax.errorbar(f_mhz[start_idx:end_idx+1], g[start_idx:end_idx+1], label=lab, **plot_kwargs)
        ax.legend()
   
    ymin, ymax = y_range 
    
    if ymax is not None:
        ax.set_ylim(top=ymax)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)

    if marker_freqs is not None:
        if not is_multi:
            series_list = [gain]
            series_labels = [None]
        else:
            series_list = gain
            series_labels = labels

        for series, series_label in zip(series_list, series_labels):
            for mf in marker_freqs:
                # Find closest index
                idx = np.argmin(np.abs(f_mhz - mf))
                marker_gain = series[idx]
                marker_freq_mhz = f_mhz[idx]

                # Plot marker
                ax.plot(marker_freq_mhz, marker_gain, 'ro', markersize=8, zorder=10)
                ann_text = f'{marker_gain:.2f} @ {mf:.0f} MHz'
                if series_label is not None:
                    ann_text = f'{series_label}: {ann_text}'
                ax.annotate(ann_text,
                            (marker_freq_mhz, marker_gain),
                            textcoords="offset points", xytext=(10, 10), ha='left',
                            fontsize=16, color='darkred')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

    if return_handles:
        return fig, ax

def plot_waterfall_heatmap_static(datetimes, spectra, faxis_mhz, title, output_path=None, show_plot=True, vmin=-80, vmax=-20,
                                  local_tz_obj=None, return_handles=False):
    """Create a heatmap of spectra with power levels as color coding. Static version with Matplotlib without interactivity.
    
    Parameters:
    -----------
    - datetimes: np.array of datetime objects. 
    - spectra: 2D np.array of shape (num_time_points, num_freq_points) containing power levels in dBm.
    - faxis_mhz: 1D np.array of frequency values in MHz corresponding to the columns of spectra.
    - output_path: Optional string path to save the plot image. If None, the plot is not saved.
    - local_tz_obj: A datetime.timezone object (e.g., timezone(timedelta(hours=-5))). 
                     If None, it will be inferred from the first datetime entry."""

    from datetime import timezone
    
    fig, ax = plt.subplots(figsize=(18, 10))

    # 1. Primary Y-axis (Local Time)
    # If no local_tz_obj is provided, we grab it from the first entry
    if local_tz_obj is None:
        local_tz_obj = datetimes[0].tzinfo

    time_hours = mdates.date2num(datetimes)

    # Clip spectra to maximum power level
    spectra_clipped = np.clip(spectra, vmin, vmax)

    # Format y-axis as time
    ax.yaxis_date()
    # Format left axis with local timezone
    local_form = mdates.DateFormatter('%d - %H:%M', tz=local_tz_obj)
    ax.yaxis.set_major_formatter(local_form)
    locator = mdates.HourLocator(interval=2)
    ax.yaxis.set_major_locator(locator)
    ax.set_ylabel(f"Time ({local_tz_obj})")

    # 2. Secondary Y-axis (GMT/UTC)
    # ax_gmt = ax.twinx()
    # ax_gmt.set_ylim(ax.get_ylim())
    # ax_gmt.yaxis_date()
    
    # Format right axis strictly as UTC
    # ax_gmt.yaxis.set_major_locator(ax.yaxis.get_major_locator())

    # gmt_form = mdates.DateFormatter('%H:%M', tz=timezone.utc)
    # ax_gmt.yaxis.set_major_formatter(gmt_form)
    # ax_gmt.set_ylabel("GMT / UTC")
    # ax_gmt.tick_params(axis='y', which='both', length=5, labelsize=15)

    # Create heatmap
    im = ax.imshow(spectra_clipped, aspect='auto', origin='upper',
                   extent=[faxis_mhz[0], faxis_mhz[-1],
                           time_hours[-1], time_hours[0]],
                   cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)

    ax.set_xlabel('Frequency (MHz)')

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dBm)')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info("Heatmap saved to %s", output_path)

    if show_plot:
        plt.show()
    else:
        plt.close()

    if return_handles:
        return fig, ax, im

# From Theo Dardio
def generate_static_hp_map(frequency_mhz, utc_timestamp, location, observer='LFSM'):
    """
    this will generate the galactic image for a specified frequency, time, and location.
    It will automatically transform the image from galactic coordinates
    to equatorial coordinates and center on the zenith (straight up)

    the "top" of the of the healpix map is the zenith (straight up)

    Parameters
    ----------
    frequency_mhz : float
        frequency in MHz
    utc_timestamp : datetime.datetime object
        UTC timestamp as a datetime object (e.g., datetime.datetime(2023, 1, 1, 12, 0, 0))
    location: tuple
        Tuple containing (latitude, longitude, elevation) of the observation site.
    observer: str
        Options: '08', '16', 'LFSM', 'Haslam'
    """
    if observer == '08':
        ov = pygdsm.GSMObserver08()
    elif observer == '16':
        ov = pygdsm.GSMObserver16()
    elif observer == 'Haslam':
        ov = pygdsm.HaslamObserver()
    elif observer == 'LFSM':
        ov = pygdsm.LFSMObserver()
    else:
        raise ValueError("Invalid observer type. Choose from '08', '16', 'LFSM', 'Haslam'.")
    
    # Set observer location and time
    lat, lon, elev = location
    ov.lon = lon
    ov.lat = lat
    ov.elev = elev
    ov.date = utc_timestamp

    hmap = ov.generate(frequency_mhz)
    hmap = np.ma.filled(hmap,fill_value=0)
    return hmap

# From Theo Dardio
def visualize_static_hmap(hmap, title="Example Galactic Healpix Map", return_handles=False):
    """
    Visualize the healpix map with azimuth and zenith angle labels.

    Parameters
    ----------
    hmap : array
        Healpix map data.
    """
        
    hp.orthview(hmap, half_sky=True, min=0, max=3000, coord='C', title=title, unit="K")

    # Now add custom labels for azimuth and zenith angle
    ax = plt.gca()

    # Add concentric zenith angle circles (like elevation rings)
    zenith_angles = [10, 30, 60, 80]
    for za in zenith_angles:
        circle = plt.Circle((0, 0), np.sin(np.radians(za)), color='white', ls='--', fill=False, alpha=0.5)
        ax.add_artist(circle)
        plt.text(0, np.sin(np.radians(za)) + 0.01, f"{za}°", color='white', ha='center')

    # Add azimuth angle labels
    az_labels = [0, 90, 180, 270]
    label_pos = {
        0: (0, 1.05),       # North (up)
        90: (1.05, 0),      # East (right)
        180: (0, -1.1),     # South (down)
        270: (-1.1, 0),     # West (left)
    }
    for az in az_labels:
        x, y = label_pos[az]
        plt.text(x, y, f"{az}°", color='white', ha='center', va='center')

    plt.show()

    if return_handles:
        return plt.gcf(), ax

# Adapted from Marcus Bosca's code
def plot_interactive_heatmap(spectra: np.ndarray, timestamps: List[datetime], mode: str = "collection",
    max_display_rows: int = 1000, max_display_cols: int = 1000, return_handles: bool = False):
    """
    Creates an interactive heatmap with dynamic downsampling and UTC time ticks.
    """
    num_rows, num_cols = spectra.shape
    
    # --- 1. Setup Figure and Normalization ---
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.20)
    
    if mode == "collection":
        norm = PowerNorm(gamma=4, vmin=-70, vmax=-50)
    else:
        norm = PowerNorm(gamma=1, vmin=-70, vmax=-35)

    # Frequency and Time Extents
    x_extent = (0.0, 409.6)
    y_extent = (0.0, float(num_rows))

    # --- 2. UI Elements (HUD) ---
    hud_text = ax.text(
        0.01, 0.995, "",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9, color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.35, pad=0.05)
    )

    # --- 3. Helper Functions ---
    def update_time_ticks(row_start: int, row_end: int):
        """Maps row indices to formatted UTC strings for the Y-axis."""
        num_ticks = 10
        if row_end <= row_start + 1:
            return
        
        # Select evenly spaced indices within the current view
        tick_rows = np.linspace(row_start, row_end - 1, num_ticks).astype(int)
        # Format the datetime objects
        tick_labels = [timestamps[r].strftime("%H:%M:%S") for r in tick_rows]
        
        ax.set_yticks(tick_rows)
        ax.set_yticklabels(tick_labels)
        ax.yaxis.set_minor_locator(NullLocator())

    def render_view(event=None):
        """Calculates which data to show based on the current zoom/pan."""
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        # Clamp limits to actual data boundaries
        x0 = max(x_extent[0], min(x_extent[1], x_lim[0]))
        x1 = max(x_extent[0], min(x_extent[1], x_lim[1]))
        y0 = max(y_extent[0], min(y_extent[1], y_lim[0]))
        y1 = max(y_extent[0], min(y_extent[1], y_lim[1]))

        # Convert coordinate limits to array indices
        col0 = int((min(x0, x1) - x_extent[0]) / (x_extent[1] - x_extent[0]) * num_cols)
        col1 = int((max(x0, x1) - x_extent[0]) / (x_extent[1] - x_extent[0]) * num_cols)
        row0 = int(min(y0, y1))
        row1 = int(max(y0, y1))

        # Slice the data for the current view
        view = spectra[row0:row1, col0:col1]
        v_rows, v_cols = view.shape
        if v_rows == 0 or v_cols == 0: return

        # Calculate downsampling steps to stay under max_display limits
        step_r = max(1, int(np.ceil(v_rows / max_display_rows)))
        step_c = max(1, int(np.ceil(v_cols / max_display_cols)))
        
        view_ds = view[::step_r, ::step_c]

        # Update image data and extent
        im.set_data(view_ds)
        im.set_extent([
            x_extent[0] + (col0 / num_cols) * (x_extent[1] - x_extent[0]),
            x_extent[0] + (col1 / num_cols) * (x_extent[1] - x_extent[0]),
            row1, # Top of view
            row0  # Bottom of view
        ])

        hud_text.set_text(f"Zoomed View: {row1-row0} rows x {col1-col0} cols | Downsample: {step_r}x")
        update_time_ticks(row0, row1)
        fig.canvas.draw_idle()

    # --- 4. Initial Plotting ---
    # Create the image object with an initial full-view downsample
    full_step_r = max(1, int(np.ceil(num_rows / max_display_rows)))
    full_step_c = max(1, int(np.ceil(num_cols / max_display_cols)))
    
    im = ax.imshow(
        spectra[::full_step_r, ::full_step_c],
        cmap='inferno',
        norm=norm,
        aspect='auto',
        extent=[x_extent[0], x_extent[1], num_rows, 0], # Note: y-axis is inverted (row 0 at top)
        interpolation="nearest",
        rasterized=True
    )

    # Visual Styling
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power (dBm)')
    ax.set_title(f'Spectrometer Data - {timestamps[0].strftime("%b %d, %Y")}')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Time (UTC)')

    # Connect the 'render_view' function to zoom/pan events
    ax.callbacks.connect('xlim_changed', render_view)
    ax.callbacks.connect('ylim_changed', render_view)

    # Initial labels
    update_time_ticks(0, num_rows)
    render_view() 
    
    plt.show()

    if return_handles:
        return fig, ax, im
