import matplotlib.pyplot as plt
import numpy as np
from .file_load import remove_spikes_from_psd
from .unit_convert import dbm_convert

LEGEND = ['6" shorted', "8' cable open",'Black body','Ambient temperature load','Noise diode',"8' cable short",'Open Circuit state']

def plot_spectrum(faxis, loaded_states, save_dir, cal_states=None, scale='log', ylabel=None, suffix='', remove_spikes=True, ymin=-75, legend=LEGEND, freq_range=None):
    """Plot the spectrum of loaded states and save the figure."""
    plt.figure(figsize=(12, 8), )
    ymax = -75
    for state in loaded_states:
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
    plt.legend(legend, fontsize=12)
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