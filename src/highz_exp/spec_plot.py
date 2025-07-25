import matplotlib.pyplot as plt
import numpy as np
from .file_load import remove_spikes_from_psd
from .unit_convert import spec_to_dbm
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

def plot_spectrum(loaded_states, save_dir, ylabel=None, suffix='', remove_spikes=True, ymin=-75, freq_range=None):
    """Plot the spectrum of loaded states and save the figure.
    
    Parameters:
        - loaded_states (dict): Dictionary of states with frequency and spectrum data, {"state_name": {"frequency": np.array, "spectrum": np.array}}. Frequency must be in MHz.
    """
    plt.figure(figsize=(12, 8), )
    ymax = -75
    for state_name, state in loaded_states.items():
        faxis = state['frequency']
        if remove_spikes:
            spectrum = remove_spikes_from_psd(faxis, state['spectrum'])
        else: spectrum = state['spectrum']
        ymax_state = max(np.max(spectrum) * 1.1, np.max(spectrum))
        if ymax_state > ymax: ymax = ymax_state
    ylim = (ymin, ymax)
    if ylabel is None:
        ylabel = 'Recorded Spectrum'
    else: ylabel=ylabel
    plt.ylim(*ylim)
    plt.xlim(*freq_range)
    plt.legend(loaded_states.keys(), fontsize=12)
    plt.ylabel(ylabel)
    plt.xlabel('Frequency [MHz]')
    plt.savefig(f'{save_dir}/calibration_states_{suffix}.png')
    