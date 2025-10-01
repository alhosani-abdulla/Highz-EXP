from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import skrf as rf
from .unit_convert import *
from .reflection_proc import impedance_from_s11
from scipy.constants import Boltzmann as k_B

def compute_spike_height_ratios(spike_data1, spike_data2, tolerance=0.1):
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

def fit_lines(y1, y2, x1=70, x2=300):
    """
    Given arrays of y-values at x1 and x2, return arrays of slope g and intercept b
    for lines y = g*x + b.

    Returns
    -------
    g : ndarray
        Slopes for each pair
    b : ndarray
        Intercepts for each pair
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    g = (y2 - y1) / (x2 - x1)
    b = y1 - g * x1
    return g, b

def load_power(V_source, Z_source, Z_load):
    """Calculate the power delivered to a load from a source voltage."""
    V_load = V_source * (Z_load / (Z_source + Z_load))
    I = V_load / Z_load
    print("Calculating power delivered to load in Watts.")
    return np.real(V_load * np.conj(I))

def johnson_voltage(T, Z, B=1):
    """Calculate the Johnson-Nyquist noise voltage, in unit of Volts/sqrt(B Hz)."""
    R = np.real(Z)
    print("Calculating Johnson-Nyquist noise voltage in Volts/sqrt(B Hz).")
    return np.sqrt(4 * k_B * T * B * R)

def apply_gain_to_power(gain_ntwk, input_power, in_offset=0, out_offset=0):
    """Return the output power (minus out_offset) as rf.ntwk object given an input power level and a gain network."""
    ratio_mag = np.abs(gain_ntwk.s[:, 0, 0])
    output_power = (input_power - in_offset) * ratio_mag - out_offset
    output_ntwk = rf.Network(s=output_power.reshape(-1, 1, 1), f=gain_ntwk.f)
    return output_ntwk

def power_delivered_from_s11(source_ntwk, load_ntwk, T_source, Z0=50, B=1):
    """Calculate the power delivered to a load from a Johnson noise source with reflection coefficient rho_source,
    load reflection coefficient rho_load, temperature T_source, and characteristic impedance Z0.

    Parameters:
    - `source_ntwk`: Network object with source reflection coefficient (S11).
    - `load_ntwk`: Network object with load reflection coefficient (S11).

    Returns a Network object with power in dBm per B Hz.
    """
    rho_source = source_ntwk.s[:, 0, 0]
    rho_load = load_ntwk.s[:, 0, 0]
    
    impd_source = impedance_from_s11(rho_source, Z0)
    impd_load = impedance_from_s11(rho_load, Z0)

    f = source_ntwk.f

    V_src = johnson_voltage(T_source, impd_source, B)
    P_transferred = load_power(V_src, impd_source, impd_load) # Power delivered to load
    P_dbm = watt_to_dbm(P_transferred)
    P_kelvin = dbm_to_kelvin(P_dbm)
    power_ntwk = rf.Network(s=P_kelvin, f=f)
    
    print("Returning Power network delivered to load in Kelvin per B Hz.")
    return power_ntwk

def fit_s11_spectrum(measured_data: rf.Network, theory_data: rf.Network, gain_func,
    extra_func, p0, n_gain_params,):
    """
    Fit the measured S11 spectrum to a model of the form:
        corrected_mag = |theory_s11 * gain_func(f, ...) + extra_func(f, ...)|

    Parameters:
    - measured_data (rf.Network)
    - theory_data (rf.Network)
    - gain_func (callable): multiplicative correction, function of form (f, *params)
    - extra_func (callable): additive correction, function of form (f, *params)
    - p0 (list): initial guess for all params (gain + extra)
    - n_gain_params (int): number of parameters in gain_func

    Returns:
    - popt (list): best-fit parameters
    - pcov (2D array): covariance matrix
    - corrected_ntwk (rf.Network): corrected theory spectrum
    """
    if not np.allclose(measured_data.f, theory_data.f):
        raise ValueError("Frequency grids must match.")

    f = measured_data.f
    f_GHz = f / 1e9  # for numerical stability

    theory_s11 = theory_data.s[:, 0, 0]
    theory_phase = np.angle(theory_s11)
    measured_mag = np.abs(measured_data.s[:, 0, 0])

    # Fit wrapper
    def fit_func(f, *params):
        gain_params = params[:n_gain_params]
        extra_params = params[n_gain_params:]
        model = theory_s11 * gain_func(f, *gain_params) + extra_func(f, *extra_params)
        return np.abs(model)

    # Fit
    popt, pcov = curve_fit(fit_func, f_GHz, measured_mag, p0=p0)

    # Apply best-fit model
    gain_params_opt = popt[:n_gain_params]
    extra_params_opt = popt[n_gain_params:]
    corrected_s11 = theory_s11 * gain_func(f_GHz, *gain_params_opt) + extra_func(f_GHz, *extra_params_opt)

    # Return new Network
    corrected_ntwk = rf.Network()
    corrected_ntwk.f = theory_data.f
    corrected_ntwk.s = corrected_s11.reshape(-1, 1, 1)
    corrected_ntwk.z0 = theory_data.z0
    corrected_ntwk.frequency = theory_data.frequency

    return popt, pcov, corrected_ntwk