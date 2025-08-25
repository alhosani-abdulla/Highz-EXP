import matplotlib.pyplot as plt
import numpy as np
from .file_load import remove_spikes_from_psd
from .unit_convert import spec_to_dbm
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

def plot_network_data(ntwk_dict, scale='linear', save_plot=True, show_phase=True, save_path=None, ylabel='Magnitude', title='Network Data', s_param=(0, 0)):
    """
    Plot magnitude (and optionally phase) from a dictionary of scikit-rf Network objects.
    Can be used for S11, gain, power spectrum, or any S-parameter data.

    Parameters:
    - ntwk_dict (dict): {label: skrf.Network}. Frequency points are in Hz.
    - scale (str): 'linear' or 'log' for magnitude scaling.
    - save_plot (bool): Whether to save the plot.
    - show_phase (bool): Whether to show phase subplot.
    - save_path (str): Path to save the plot.
    - ylabel (str): Y-axis label for magnitude plot.
    - title (str): Plot title.
    - s_param (tuple): S-parameter indices (i, j) to plot. Default (0, 0) for S11.
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
        s_data = ntwk.s[:, s_param[0], s_param[1]]

        magnitude = 20 * np.log10(np.abs(s_data)) if scale == 'log' else np.abs(s_data)
        phase = np.angle(s_data, deg=True)

        color = color_cycle[idx % len(color_cycle)]

        ax_mag.plot(freq / 1e6, magnitude, label=f'{label}', color=color)
        if show_phase:
            ax_phase.plot(freq / 1e6, phase, label=f'{label}', color=color, linestyle='--')

    ax_mag.set_ylabel(ylabel, fontsize=14)
    ax_mag.grid(True)
    ax_mag.legend(loc='best', fontsize=12)

    if show_phase:
        ax_phase.set_xlabel('Frequency [MHz]', fontsize=14)
        ax_phase.set_ylabel('Phase [deg]', fontsize=14)
        ax_phase.grid(True)
        ax_phase.legend(loc='best', fontsize=12)

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
    
def plot_smith_chart(ntwk_dict, suffix='LNA', save_plot=True, save_dir=None, legend_loc='best', individual=True):
    """
    Plot Smith chart from one or more scikit-rf Network objects.

    Parameters:
    - ntwk_dict (dict): {label: rf.Network} pairs.
    - suffix (str): Used for output filename if saving.
    - legend_loc (str): Location of the legend. Default is 'best'.
    - individual (bool): Whether to plot/save individual Smith charts for each network.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)

    for label, ntwk in ntwk_dict.items():
        ntwk.plot_s_smith(ax=ax, label=label, chart_type='z', draw_labels=True, label_axes=True)

    ax.set_title(f'{suffix} Smith Chart', fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0., fontsize=12)
    plt.tight_layout()

    if save_plot:
        suffix = suffix.replace(' ', '_')
        # Save to current directory if no path info is available
        if save_dir is None:
            save_dir = os.getcwd()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(pjoin(save_dir, f'{suffix}_smith_chart.png'), bbox_inches='tight')

    plt.show()
    if individual:
        for label, ntwk in ntwk_dict.items():
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)
            ntwk.plot_s_smith(ax=ax, label=label, chart_type='z', draw_labels=True, label_axes=True)
            ax.set_title(f'Smith Chart: {label}', fontsize=16)
            if save_plot:
                label_safe = label.replace(' ', '_')
                outname = f'{suffix}_smith_{label_safe}.png'
                fig.savefig(outname, bbox_inches='tight')

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

def plot_spectrum(loaded_states_ntwk, save_dir, ylabel=None, suffix='', ymin=-75, freq_range=None):
    """Plot the spectrum from a dictionary of scikit-rf Network objects and save the figure.
    
    Parameters:
        - ymin (float): Minimum y-axis value
        - freq_range (tuple, optional): Frequency range to plot (fmin, fmax) in MHz
        - s_param (tuple): S-parameter indices (i, j) to plot. Default (0, 0) for S11.
    """
    plt.figure(figsize=(12, 8))
    ymax = ymin
    
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for idx, (state_name, ntwk) in enumerate(loaded_states_ntwk.items()):
        freq = ntwk.f  # in Hz
        spectrum = np.real(ntwk.s[:, 0, 0])
        
        # Convert frequency to MHz for plotting
        faxis_mhz = freq / 1e6
        
        color = color_cycle[idx % len(color_cycle)]
        plt.plot(faxis_mhz, spectrum, label=state_name, color=color)
        
        ymax_state = np.max(spectrum)
        if ymax_state > ymax: 
            ymax = ymax_state + 10
    
    # Adjust ymax with some padding
    
    ylim = (ymin, ymax)
    if ylabel is None:
        ylabel = 'PSD [dBm]'

    plt.ylim(*ylim)
    if freq_range is not None:
        plt.xlim(*freq_range)
    plt.legend(fontsize=12)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel('Frequency [MHz]', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/spectrum_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    