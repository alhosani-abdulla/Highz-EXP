import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from os.path import join as pjoin, basename as pbase

def load_s1p(s1p_files, labels=None) -> dict:
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

def fit_reflection_coeff(s1p_ntwk, guess_A_real, guess_A_imag, guess_delay, save_path=None) -> tuple[complex, float, rf.Network]:
    """Fit the reflection coefficient from an S1P network to a model of the form (A_real + 1j*A_imag) * exp(j*2*pi*f*t_delay).

    Returns
    - A_fitted (complex): Fitted amplitude (complex).
    - t_delay_fitted (float): Fitted time delay.
    - s1p_fitted_ntwk (rf.Network): Fitted reflection coefficient as a Network object.
    """
    f = s1p_ntwk.f
    gamma_meas = s1p_ntwk.s[:, 0, 0]
    gamma_meas_comb = np.concatenate([gamma_meas.real, gamma_meas.imag])

    def reflection_model(freq, A_real, A_imag, t_delay):
        A = A_real + 1j * A_imag
        model = A * np.exp(1j * 2 * np.pi * freq * t_delay)
        return np.concatenate([model.real, model.imag])

    popt, pcov = curve_fit(
        reflection_model, f, gamma_meas_comb, p0=[guess_A_real, guess_A_imag, guess_delay]
    )
    A_fitted = popt[0] + 1j * popt[1]
    t_delay_fitted = popt[2]
    s1p_fitted = A_fitted * np.exp(1j * 2 * np.pi * f * t_delay_fitted)
    # Create a new Network object with the same frequency and fitted S-parameters
    s1p_fitted_ntwk = rf.Network(s=s1p_fitted, f=f)
    if save_path is not None:
        s1p_fitted_ntwk.write_touchstone(save_path, overwrite=True)
    return A_fitted, t_delay_fitted, s1p_fitted_ntwk

def plot_s11_reflect(ntwk_dict, scale='linear', save_plot=True, show_phase=True, save_path=None):
    """
    Plot S11 magnitude (and optionally phase) from a dictionary of scikit-rf Network objects.

    Parameters:
    - ntwk_dict (dict): {label: skrf.Network}
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
    
def plot_smith_chart(s1p_files, suffix='LNA', labels=None, save_plot=True, legend_loc='best', individual=True):
    """
    Plot Smith chart from one or more S1P files.

    Parameters:
    - s1p_files (list of str): Paths to .s1p files.
    - suffix (str): Used for output filename if saving.
    - labels (list of str, optional): Labels for legend.
    - save_plot (bool): Whether to save the figure.
    - legend_loc (str): Location of the legend. Default is 'best'.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)

    if labels is None:
        labels = [pbase(f) for f in s1p_files]

    for file, label in zip(s1p_files, labels):
        ntwk = rf.Network(file)
        ntwk.plot_s_smith(ax=ax, label=label, chart_type='z', draw_labels=True, label_axes=True)

    ax.set_title(f'{suffix} Smith Chart')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
    plt.tight_layout()

    if save_plot:
        suffix = suffix.replace(' ', '_')
        fig.savefig(pjoin(os.path.dirname(s1p_files[0]), f'{suffix}_smith_chart.png'))

    plt.show()
    if individual:
        for file, label in zip(s1p_files, labels):
            ntwk = rf.Network(file)
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)
            ntwk.plot_s_smith(ax=ax, label=label, chart_type='z', draw_labels=True, label_axes=True)

            ax.set_title(f'Smith Chart: {label}')

            if save_plot:
                label = label.replace(' ', '_')
                outname = f'{suffix}_smith_{label}.png'
                fig.savefig(pjoin(os.path.dirname(file), outname), bbox_inches='tight')