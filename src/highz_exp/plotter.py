import matplotlib.pyplot as plt
import numpy as np
from .spec_class import Spectrum
from os.path import join as pjoin, basename as pbase
import os
import skrf as rf

LEGEND = ['6" shorted', "8' cable open",'Black body','Ambient temperature load','Noise diode',"8' cable short",'Open Circuit state']

def plot_measured_vs_fitted(ntwk_dict, scale='linear', save_plot=True, save_path=None, ylabel='Magnitude', title='Measured vs Fitted Spectrum', show_residual=False, show_bottom_panel=True):
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
    ax_mag.set_ylabel(ylabel, fontsize=14)
    ax_mag.legend(loc='best')
    ax_mag.grid(True)

    if show_bottom_panel:
        ax_bottom = axes[1]
        
        if show_residual:
            residual = mag_measured - mag_fitted
            ax_bottom.plot(freq / 1e6, residual, color='C2')
            ax_bottom.axhline(0, color='red', linestyle='-', linewidth=1.5, label='residual = 0')
            ax_bottom.set_ylabel('Residual (Measured - Theory)', fontsize=14)
        else:
            ratio = mag_measured / mag_fitted
            ax_bottom.plot(freq / 1e6, 1 / ratio, color='C2')
            ax_bottom.axhline(1, color='red', linestyle='-', linewidth=1.5, label='measured/theory = 1')
            ax_bottom.set_ylabel('Measured/Theory', fontsize=14)
        
        ax_bottom.set_xlabel('Frequency [MHz]', fontsize=14)
        ax_bottom.grid(True)
        ax_bottom.legend(loc='best')
    else:
        ax_mag.set_xlabel('Frequency [MHz]', fontsize=14)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plot:
        if save_path is not None:
            plt.savefig(save_path)
        else:
            print("! Save path not entered.")

    plt.show()

def plot_network_data(ntwk_dict, save_plot=True, show_phase=True, save_path=None, ylabel='Magnitude', title='Network Data', s_param=(0, 0),
                      ylim=None):
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

    ax_mag.set_ylabel(ylabel, fontsize=20)
    ax_mag.grid(True)
    ax_mag.legend(loc='best', fontsize=20)
    ax_mag.tick_params(axis='both', labelsize=20, which='major')

    if ylim is not None:
        ax_mag.set_ylim(ylim)

    if show_phase:
        ax_phase.set_xlabel('Frequency [MHz]', fontsize=20)
        ax_phase.set_ylabel('Phase [deg]', fontsize=18)
        ax_phase.grid(True)
        ax_phase.legend(loc='best', fontsize=18)

    ax_mag.set_xlabel('Frequency [MHz]', fontsize=20)

    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plot:
        if save_path is not None:
            plt.savefig(save_path)
        else:
            print("! Save path not entered.")

    plt.show()


def plot_s1p(ntwk_dict, db=True, title='Reflection Measurement (S11)', ymax=None, ymin=None, show_phase=False, attenuation=0, save_dir=None, suffix=None):
    """
    Plot multiple reflections from .s1p Network objects on the same axes.

    Parameters:
    - ntwk_dict (dict): {'label': rf.Network}
    - db (bool): If True, plot reflection in dB
    - show_phase (bool): If True, also plot phase in degrees (dashed lines)
    - attenuation (float): Attenuation added to the magnitude (dB)
    - save_dir (str): directory to save the combined plot
    - suffix (str): optional suffix for saved filename

    Returns:
    - dict: the same ntwk_dict passed in
    """
    if not ntwk_dict:
        print("No networks provided.")
        return ntwk_dict

    fig, ax1 = plt.subplots(figsize=(14, 8))
    if show_phase:
        # replace the single axis with two stacked axes (top: magnitude, bottom: phase)
        fig.clf()
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
    else:
        ax2 = None

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (label, network) in enumerate(ntwk_dict.items()):
        freq = network.f
        s11 = network.s[:, 0, 0]
        mag = 20 * np.log10(np.abs(s11)) + attenuation if db else np.abs(s11)
        phase = np.angle(s11, deg=True)

        color = color_cycle[idx % len(color_cycle)]
        ax1.plot(freq / 1e6, mag, label=f'{label}', color=color)
        if show_phase:
            ax2.plot(freq / 1e6, phase, color=color, linestyle='--', label=f'{label} (phase)')

    ax1.set_xlabel('Frequency [MHz]', fontsize=20)
    ax1.set_ylabel('Reflection' + (' [dB]' if db else ''), fontsize=20)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    if ymax is not None:
        ax1.set_ylim(top=ymax)
    if ymin is not None:
        ax1.set_ylim(bottom=ymin)

    if show_phase:
        ax2.set_ylabel('Phase [deg]', color='r', fontsize=18)
        ax2.tick_params(axis='y', labelsize=16)

        # Combine legends from both axes
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=18, loc='best')
    else:
        ax1.legend(loc='best', fontsize=18)

    plt.title(title, fontsize=22)
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        safe_suffix = f"_{suffix}" if suffix else ""
        plt.savefig(f'{save_dir}/Reflection{safe_suffix}.png', dpi=150, bbox_inches='tight')

    plt.show()
    return ntwk_dict

def plot_load_s2p(file_path, db=True, x_scale='linear', title='Gain Measurement (S21)', ymax=None, ymin=None, show_phase=False, attenuation=0, save_dir=None, suffix=None) -> rf.Network:
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
    ax1.set_xlabel('Frequency [MHz]', fontsize=20)
    ax1.set_ylabel('Gain' + (' [dB]' if db else ''), fontsize=20)
    if ymax is not None:
        ax1.set_ylim(top=ymax)
    if ymin is not None:
        ax1.set_ylim(bottom=ymin)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=18)

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
    if show_phase:
        ax2 = ax1.twinx()
        ax2.plot(freq / 1e9, phase, color='r', linestyle='--', label='Phase [deg]')
        ax2.set_ylabel('Phase [deg]', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

    plt.title(title, fontsize=22)
    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(f'{save_dir}/Gain_{suffix}.png')
    plt.show()

    return network

def plot_spectrum(loaded_specs:list[Spectrum], save_dir=None, ylabel=None, suffix='', ymin=-75, ymax=None, freq_range=None, yticks=None, title='Recorded Spectrum', show_plot=True):
    """Plot the spectrum from a dictionary of scikit-rf Network objects and save the figure if save_dir is not None.
    
    Parameters:
        - loaded_specs: list of Spectrum objects to plot, with frequency in Hz.
        - ymin (float): Minimum y-axis value
        - freq_range (tuple, optional): Frequency range to plot (fmin, fmax) in MHz
        - s_param (tuple): S-parameter indices (i, j) to plot. Default (0, 0) for S11.
    """
    plt.figure(figsize=(14, 8))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if ymax is None:
        ymax = ymin  # Initialize with ymin

        for idx, spec in enumerate(loaded_specs):
            freq = spec.freq  # in Hz
            spectrum = spec.spec
            
            # Convert frequency to MHz for plotting
            faxis_mhz = freq / 1e6
            
            color = color_cycle[idx % len(color_cycle)]
            plt.plot(faxis_mhz, spectrum, label=spec.name, color=color)
            
            ymax_state = np.max(spectrum)
            if ymax_state > ymax: 
                ymax = ymax_state
        
        # Adjust ymax with some padding
    else:
        for idx, spec in enumerate(loaded_specs):
            freq = spec.freq  # in Hz
            spectrum = spec.spec
            
            # Convert frequency to MHz for plotting
            faxis_mhz = freq / 1e6
            
            color = color_cycle[idx % len(color_cycle)]
            plt.plot(faxis_mhz, spectrum, label=spec.name, color=color)

    ylim = (ymin, ymax)
    if ylabel is None:
        ylabel = 'PSD [dBm]'

    plt.ylim(*ylim)
    if freq_range is not None:
        plt.xlim(*freq_range)
    plt.legend(fontsize=18, ncol=2, loc='best')
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel('Frequency [MHz]', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    if yticks is not None:
        plt.yticks(yticks)
    plt.title(title, fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/spectrum_{suffix}.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()

def plot_gain(f, gain, label=None, start_freq=10, end_freq=400, ymax=None, xlabel='Frequency (MHz)', ylabel='Gain (dB)', title=None, save_path=None):
    """Plot gain over a specified frequency range.
    
    Parameters:
        - f (np.ndarray): Frequency axis in MHz.
        - gain (np.ndarray or list of np.ndarray): Gain values in dB."""
    # Find the index closest to start_freq and end_freq
    start_idx = np.argmin(np.abs(f - start_freq))
    end_idx = np.argmin(np.abs(f - end_freq))
    plt.figure(figsize=(12, 8))
    if not isinstance(gain, list):
        plt.plot(f[start_idx:end_idx+1], gain[start_idx:end_idx+1])
    else:
        for g, lab in zip(gain, label):
            plt.plot(f[start_idx:end_idx+1], g[start_idx:end_idx+1], label=lab)
        plt.legend(fontsize=18)
    
    if ymax is not None:
        plt.ylim(top=ymax)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.title(title, fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


