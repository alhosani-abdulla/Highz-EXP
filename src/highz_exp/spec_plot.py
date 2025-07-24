import matplotlib.pyplot as plt
import numpy as np
from .file_load import remove_spikes_from_psd
from .unit_convert import dbm_convert
from os.path import join as pjoin
import os
import skrf as rf

LEGEND = ['6" shorted', "8' cable open",'Black body','Ambient temperature load','Noise diode',"8' cable short",'Open Circuit state']

def plot_s2p_gain(file_path, db=True, x_scale='linear', title='Gain Measurement (S21)', show_phase=False, attenuation=0, save_plot=True, save_name='S21_Measurement'):
    """
    Load and plot gain (S21) from an S2P file.

    Parameters:
    - file_path (str/list): Path to the .s2p file
    - db (bool): If True, plot gain in dB
    - show_phase (bool): If True, also plot phase in degrees
    - attenuation (float): Attenuation that was applied to the gain measurements
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
    fig.set_size_inches(10, 6)
    if x_scale == 'log':
        ax1.set_xscale('log')
    ax1.plot(freq / 1e6, mag, color='b', label='Gain (S21)' + (' [dB]' if db else ''), alpha=0.5)
    ax1.set_xlabel('Frequency [MHz]')
    ax1.set_ylabel('Gain' + (' [dB]' if db else ''), color='b')
    ax1.set_ylim(top=1.3 * np.max(mag))
    ax1.grid(True)
    ax1.tick_params(axis='y', labelcolor='b')

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
                    fontsize=9, color='darkred')
    if show_phase:
        ax2 = ax1.twinx()
        ax2.plot(freq / 1e9, phase, color='r', linestyle='--', label='Phase [deg]')
        ax2.set_ylabel('Phase [deg]', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

    plt.title(title)
    fig.tight_layout()
    if save_plot:
        plt.savefig(pjoin(parent_dir, f'{save_name}.png'))
    plt.show()

    return network

def plot_spectrum(loaded_states, save_dir, cal_states=None, scale='log', ylabel=None, suffix='', remove_spikes=True, ymin=-75, legend=LEGEND, freq_range=None):
    """Plot the spectrum of loaded states and save the figure.
    
    Parameters:
    - loaded_states (dict): Dictionary of states with frequency and spectrum data, {"state_name": {"frequency": np.array, "spectrum": np.array}}
    - scale (str): 'log' or 'linear' for the y-axis scale
    """
    plt.figure(figsize=(12, 8), )
    ymax = -75
    for state_name, state in loaded_states.items():
        faxis = state['frequency']
        if remove_spikes:
            spectrum = remove_spikes_from_psd(faxis, state['spectrum'])
        else: spectrum = state['spectrum']
        if scale == 'log':
            plt.plot(faxis, dbm_convert(spectrum))
            ymax_state = max(np.max(dbm_convert(spectrum)) * 0.8, np.max(dbm_convert(spectrum)))
        elif scale == 'linear':
            plt.plot(faxis, spectrum)
            ymax_state = max(np.max(spectrum) * 1.1, np.max(spectrum))
        else:
            raise ValueError("scale must be linear or log.")
        if ymax_state > ymax: ymax = ymax_state
    ylim = (ymin, ymax)
    if ylabel is None:
        ylabel = 'Power Spectral Density [dBm]' if scale == 'log' else r'Power Spectral Density'
    else: ylabel=ylabel
    plt.ylim(*ylim)
    plt.xlim(*freq_range)
    plt.legend(loaded_states.keys(), fontsize=12)
    plt.ylabel(ylabel)
    plt.xlabel('Frequency [MHz]')
    plt.savefig(f'{save_dir}/calibration_states_{scale}{suffix}.png')
    
def plot_s11_reflect(ntwk_dict, scale='linear', save_plot=True, show_phase=True, save_path=None):
    """
    Plot S11 magnitude (and optionally phase) from a dictionary of scikit-rf Network objects.

    Parameters:
    - ntwk_dict (dict): {label: skrf.Network}
    - scale (str): 'linear' or 'log' for magnitude scale
    - save_plot (bool): If True, save the plot
    - show_phase (bool): If True, plot phase in degrees
    - save_path (str): File path to save the plot
    """

    nrows = 2 if show_phase else 1
    fig, axes = plt.subplots(nrows=nrows, figsize=(10, 6), sharex=True)
    if nrows == 1:
        axes = [axes]  # Make it iterable for consistency

    ax_mag = axes[0]
    ax_phase = axes[1] if show_phase else None

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (label, ntwk) in enumerate(ntwk_dict.items()):
        freq = ntwk.f  # in Hz
        s11 = ntwk.s[:, 0, 0]

        magnitude = 20 * np.log10(np.abs(s11)) if scale == 'log' else np.abs(s11)
        phase = np.angle(s11, deg=True)

        color = color_cycle[idx % len(color_cycle)]

        ax_mag.plot(freq / 1e6, magnitude, label=f'{label} |S11|', color=color)
        if show_phase:
            ax_phase.plot(freq / 1e6, phase, label=f'{label} ∠S11', color=color, linestyle='--')

    # Format magnitude plot
    ax_mag.set_ylabel('Magnitude [dB]' if scale == 'log' else 'Magnitude')
    ax_mag.grid(True)
    ax_mag.legend(loc='best')

    # Format phase plot
    if show_phase:
        ax_phase.set_xlabel('Frequency [MHz]')
        ax_phase.set_ylabel('Phase [deg]')
        ax_phase.grid(True)
        ax_phase.legend(loc='best')

    else:
        ax_mag.set_xlabel('Frequency [MHz]')

    fig.suptitle('S11 Reflection Coefficient')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plot:
        if save_path is not None:
            plt.savefig(save_path)
        else:
            print("! Save path not entered.")

    plt.show()