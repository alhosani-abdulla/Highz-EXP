import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin, basename as pbase

def load_file(s1p_files, labels=None):
    """
    Load S1P files and return a dictionary of label: rf.Network objects.

    Parameters:
    - s1p_files (list of str): Paths to .s1p files.
    - labels (list of str, optional): Labels for the files.

    Returns:
    - dict: {label: rf.Network}
    """
    if labels is None:
        labels = [pbase(f) for f in s1p_files]

    return {label: rf.Network(file) for file, label in zip(s1p_files, labels)}

def LNA_total_reflection(rho_cable_ntwk, rho_LNA_ntwk):
    """Return the total reflection coefficient with multiple reflections between cable and LNA interfaces"""
    rho_cable = rho_cable_ntwk.s[:, 0, 0]
    rho_LNA = rho_LNA_ntwk.s[:, 0, 0]
    a_lna = rho_cable * 1 / (1 - rho_cable*rho_LNA)
    a_lna_ntwk = rf.Network(s=a_lna, f=rho_cable_ntwk.f)
    return a_lna_ntwk

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