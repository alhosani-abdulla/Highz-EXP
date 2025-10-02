import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from os.path import join as pjoin, basename as pbase

def compute_spike_height_ratios(spike_data1, spike_data2, tolerance=0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the ratio of spike heights between two datasets with tolerance matching.

    Parameters:
    -----------
    spike_data1 : np.ndarray
        2D array with shape (n_spikes, 2) containing [x_vals, heights]
    spike_data2 : np.ndarray
        2D array with shape (m_spikes, 2) containing [x_vals, heights]
    tolerance : float
        Maximum absolute difference in x_vals to consider as a match

    Returns:
    --------
    ratios : np.ndarray
        Array of ratios (spike_data1_height / spike_data2_height) for matched spikes
    matched_x1 : np.ndarray
        x_vals from spike_data1 that were matched
    matched_x2 : np.ndarray
        x_vals from spike_data2 that were matched
    """
    x1, h1 = spike_data1[:, 0], spike_data1[:, 1]
    x2, h2 = spike_data2[:, 0], spike_data2[:, 1]

    ratios = []
    matched_x1 = []
    matched_x2 = []

    # For each spike in dataset 1, find closest match in dataset 2
    for i in range(len(x1)):
        # Calculate absolute differences
        diffs = np.abs(x2 - x1[i])
        min_idx = np.argmin(diffs)
        min_diff = diffs[min_idx]

        # Check if within tolerance
        if min_diff <= tolerance:
            # Avoid division by zero
            if h2[min_idx] != 0:
                ratios.append(h1[i] / h2[min_idx])
                matched_x1.append(x1[i])
                matched_x2.append(x2[min_idx])

    # Print results
    print("Matched spikes:")
    print(f"{'x1':>8} {'x2':>8} {'h1/h2':>8}")
    print("-" * 26)
    for x1_val, x2_val, r in zip(matched_x1, matched_x2, ratios):
        print(f"{x1_val:8.2f} {x2_val:8.2f} {r:8.2f}")

    return np.array(ratios), np.array(matched_x1), np.array(matched_x2)

def LNA_total_reflection(rho_cable_ntwk, rho_LNA_ntwk):
    """Return the total reflection coefficient with multiple reflections between cable and LNA interfaces"""
    rho_cable = rho_cable_ntwk.s[:, 0, 0]
    rho_LNA = rho_LNA_ntwk.s[:, 0, 0]
    a_lna = rho_cable * 1 / (1 - rho_cable*rho_LNA)
    a_lna_ntwk = rf.Network(s=a_lna, f=rho_cable_ntwk.f)
    return a_lna_ntwk

def s11_reflected_power(s11_db, p_in_k=50):
    """Compute reflected power (in Kelvin) given S11 in dB and input power.
    
    Parameters:
    - s11_db: S11 in dB (can be float or NumPy array)
    - p_in_k: Input power in Kelvin (default is 50 K)

    Returns:
    - Reflected power in Kelvin
    """
    R = 10 ** (s11_db / 10)  # power reflection coefficient
    return p_in_k * R

def impedance_from_s11(rho, Z0=50):
    """Convert reflection coefficient to impedance."""
    return Z0 * (1+rho)/(1-rho)

def fit_exp_decay(s1p_ntwk, guess_A_real, guess_A_imag, guess_delay, guess_alpha, save_path=None) -> tuple[complex, float, rf.Network]:
    """Fit the reflection coefficient from an S1P network to a model of the form (A_real + 1j*A_imag) * exp(j*2*pi*f*t_delay).

    Returns
    - A_fitted (complex): Fitted amplitude (complex).
    - t_delay_fitted (float): Fitted time delay.
    - s1p_fitted_ntwk (rf.Network): Fitted reflection coefficient as a Network object.
    """
    f = s1p_ntwk.f
    gamma_meas = s1p_ntwk.s[:, 0, 0]
    gamma_meas_comb = np.concatenate([gamma_meas.real, gamma_meas.imag])

    def reflection_model(freq, A_real, A_imag, t_delay, alpha):
        A = A_real + 1j * A_imag
        decay = np.exp(-alpha * freq)
        model = decay * A * np.exp(1j * 2 * np.pi * freq * t_delay)
        return np.concatenate([model.real, model.imag])

    popt, pcov = curve_fit(
        reflection_model, f, gamma_meas_comb, p0=[guess_A_real, guess_A_imag, guess_delay, guess_alpha]
    )
    A_fitted = popt[0] + 1j * popt[1]
    t_delay_fitted = popt[2]
    alpha_fitted = popt[3]
    s1p_fitted = A_fitted * np.exp(-alpha_fitted * f) * np.exp(1j * 2 * np.pi * f * t_delay_fitted)
    # Create a new Network object with the same frequency and fitted S-parameters
    s1p_fitted_ntwk = rf.Network(s=s1p_fitted, f=f)
    if save_path is not None:
        s1p_fitted_ntwk.write_touchstone(save_path, overwrite=True)
    return A_fitted, t_delay_fitted, alpha_fitted, s1p_fitted_ntwk

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

