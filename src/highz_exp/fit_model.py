from os.path import join as pjoin
import numpy as np
from scipy.optimize import curve_fit
import skrf as rf
from .unit_convert import *
from .reflection_proc import impedance_from_s11
from scipy.constants import Boltzmann as k_B
from highz_exp.spec_class import Spectrum
from matplotlib import pyplot as plt

# ==============================================================================
# Generic Polynomial Fitting (kept as standalone)
# ==============================================================================
# fit_polynomial_metric() is a flexible function for fitting any frequency-dependent
# metric (gain, temperature, noise figure, etc.) with an Nth-order polynomial.
#
# Examples:
#   from highz_exp.fit_model import fit_polynomial_metric
#
#   # Fit gain with 4th-order polynomial
#   result_gain = fit_polynomial_metric(freq, gain_db, order=4, metric_name='gain')
#
#   # Fit temperature with 3rd-order polynomial  
#   result_temp = fit_polynomial_metric(freq, temp_k, order=3, metric_name='temperature')
#
#   # Fit arbitrary metric with 2nd-order polynomial using scipy backend
#   result_custom = fit_polynomial_metric(
#       freq, my_metric, order=2, metric_name='custom_metric', method='scipy'
#   )
#
#   # All results include: coefficients, order, fitted, residuals, r2, covariance, valid_mask
#   # Plus convenience keys 'a0', 'a1', ..., 'aN' for individual coefficients
#
# ==============================================================================

def _poly_model(freq, *coeffs):
    """Evaluate polynomial at freq using coefficients in ascending order."""
    coeffs = np.asarray(coeffs, dtype=float)
    return np.polynomial.polynomial.polyval(freq, coeffs)

def _poly_inv_model(freq, *coeffs):
    """Evaluate polynomial + 1/f term at freq."""
    coeffs = np.asarray(coeffs, dtype=float)
    a_inv = coeffs[0]
    poly_coeffs = coeffs[1:]
    return a_inv / freq + np.polynomial.polynomial.polyval(freq, poly_coeffs)

def fit_polynomial_metric(frequency, metric, order=4, metric_name='metric', 
                          method='numpy', initial_guess=None, include_inv_f=False):
    """Generic function to fit any frequency-dependent metric with an Nth-order polynomial.
    
    Fits: metric(f) = a0 + a1*f + a2*f^2 + ... + aN*f^N
    
    Or if include_inv_f=True:
         metric(f) = a_inv/f + a0 + a1*f + a2*f^2 + ... + aN*f^N
    
    Parameters
    ----------
    frequency : np.ndarray
        Frequency axis (any units).
    metric : np.ndarray
        Metric values to fit (gain, temperature, etc.).
    order : int, optional
        Polynomial order (default: 4).
    metric_name : str, optional
        Name of metric for error messages and reporting (default: 'metric').
    method : str, optional
        Fit backend: 'numpy' (polyfit) or 'scipy' (curve_fit). Default: 'numpy'.
        When include_inv_f=True, scipy is automatically used.
    initial_guess : tuple[float, ...], optional
        Initial guess for scipy fit: (a0, a1, ..., aN).
        Must have length order + 1. If None, uses polyfit result.
        When include_inv_f=True, should be (a_inv, a0, a1, ..., aN) with length order + 2.
    include_inv_f : bool, optional
        If True, include a 1/f term in the model. Default: False.
        When True, scipy method is automatically used.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'coefficients': np.ndarray [a0, a1, ..., aN] in ascending order
          (or [a_inv, a0, a1, ..., aN] if include_inv_f=True)
        - 'order': polynomial order
        - 'fitted': fitted values at valid frequency points
        - 'residuals': metric - fitted at valid points
        - 'r2': coefficient of determination
        - 'covariance': covariance matrix (None for method='numpy')
        - 'valid_mask': boolean mask of valid (finite) points used in fit
        - 'a0', 'a1', ..., 'aN': individual coefficient values (convenience keys)
        - 'a_inv': 1/f coefficient (if include_inv_f=True)
        - 'include_inv_f': boolean indicating if 1/f term was included
    """
    frequency = np.asarray(frequency, dtype=float)
    metric = np.asarray(metric, dtype=float)
    
    if not isinstance(order, int) or order < 0:
        raise ValueError('order must be a non-negative integer.')
    
    valid = np.isfinite(frequency) & np.isfinite(metric)
    f = frequency[valid]
    m = metric[valid]
    
    min_samples = order + 2 if include_inv_f else order + 1
    if f.size < min_samples:
        raise ValueError(f'Need at least {min_samples} valid samples for order={order} '
                        f'polynomial fit (include_inv_f={include_inv_f}) of {metric_name}.')
    
    method = method.lower() if method else 'numpy'
    
    if include_inv_f:
        # Force scipy for 1/f term
        if initial_guess is None:
            # Rough initial guess: start with polyfit and set a_inv to 0
            poly_desc = np.polyfit(f, m, deg=order)
            poly_asc = poly_desc[::-1]
            initial_guess = np.concatenate([[0.0], poly_asc])
        
        initial_guess = np.asarray(initial_guess, dtype=float)
        if initial_guess.size != (order + 2):
            raise ValueError(f'When include_inv_f=True, initial_guess must have length '
                           f'order + 2 = {order + 2}.')
        
        coefficients, covariance = curve_fit(
            _poly_inv_model,
            f,
            m,
            p0=initial_guess,
            maxfev=50000,
        )
        fitted = _poly_inv_model(f, *coefficients)
    elif method == 'numpy':
        # np.polyfit returns descending order: [aN, ..., a1, a0]
        poly_desc = np.polyfit(f, m, deg=order)
        coefficients = poly_desc[::-1]  # reverse to ascending: [a0, ..., aN]
        covariance = None
        fitted = _poly_model(f, *coefficients)
    elif method == 'scipy':
        if initial_guess is None:
            poly_desc = np.polyfit(f, m, deg=order)
            initial_guess = poly_desc[::-1]
        
        initial_guess = np.asarray(initial_guess, dtype=float)
        if initial_guess.size != (order + 1):
            raise ValueError(f'initial_guess must have length order + 1 = {order + 1}.')
        
        coefficients, covariance = curve_fit(
            _poly_model,
            f,
            m,
            p0=initial_guess,
            maxfev=50000,
        )
        fitted = _poly_model(f, *coefficients)
    else:
        raise ValueError("method must be 'numpy' or 'scipy'.")
    
    residuals = m - fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((m - np.mean(m))**2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    
    result = {
        'coefficients': np.asarray(coefficients, dtype=float),
        'order': int(order),
        'fitted': fitted,
        'residuals': residuals,
        'r2': float(r2),
        'covariance': covariance,
        'valid_mask': valid,
    }
    
    return result


class CALModel:
    """Class for fitting calibrator metrics with polynomial models."""
    @staticmethod
    def fit_and_plot(f, T, order=1, initial_guess=None):
        """Fit calibrator temperature with a polynomial model and plot the results.
        
        Return: 
        dict with keys 'coefficients', 'order', 'fitted', 'residuals', 'r2', 'covariance', 'valid_mask'"""
        fitted = CALModel.fit_temperature(f, T, order, initial_guess)

        valid = fitted['valid_mask']
        f_valid_hz = f[valid]
        t_valid = T[valid]
        t_model = fitted['fitted']

        plt.figure(figsize=(10, 6))
        plt.plot(f_valid_hz / 1e6, t_valid, 'o', ms=4, alpha=0.8, label='Measured')
        plt.plot(f_valid_hz / 1e6, t_model, '-', lw=2, label=f'fit')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Temperature (K)')
        plt.title('Calibrator Temperature Fit')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return fitted
    
    @staticmethod
    def model_eval(f, coeffs):
        """Evaluate linear noise diode temperature model: T(f) = polyval(f, coeffs)."""
        coeffs = np.asarray(coeffs, dtype=float)
        return np.polynomial.polynomial.polyval(f, coeffs)
    
    @staticmethod
    def fit_temperature(f, T, order=1, initial_guess=None):
        """Fit noise diode temperature with a polynomial model. Default: a0 + a1*f
        
        Parameters
        ----------
        order : int, optional
            Polynomial order (default: 1).
        initial_guess : tuple[float, ...], optional
            Initial guess for scipy fit: (a0, a1, ..., aN).
            Must have length order + 1.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'coefficients' (np.ndarray: [a0, a1, ..., aN])
            - 'order', 'fitted', 'residuals', 'r2', 'covariance', 'valid_mask'
            Also includes convenience keys 'a0', 'a1', ..., 'aN'."""
        return fit_polynomial_metric(f, T, order=order, metric_name='Calibrator Temperature',
            method='numpy', initial_guess=initial_guess, include_inv_f=False
        )
    

# ==============================================================================
# LNA Model Class
# ==============================================================================

class LNAModel:
    """Class for fitting and modeling Low Noise Amplifier (LNA) metrics.
    
    This class encapsulates LNA-specific fitting routines and physics calculations
    for temperature, gain, and other LNA metrics.
    """
    @staticmethod
    def model_eval(f, a0, a1, b, a2, a3):
        """Evaluate LNA temperature model: T(f) = a0/f^2 + a1/(f+b) + a2 + a3*f."""
        f = np.asarray(f, dtype=float)
        return a0 / (f**2) + a1 / (f + b) + a2 + a3 * f
    
    @staticmethod
    def fit_temperature(frequency, temperature, method='scipy', initial_guess=None):
        """Fit LNA temperature.

        Fits:
            T(f) = a0/f^2 + a1/(f+b) + a2 + a3*f

        Parameters
        ----------
        frequency : np.ndarray
            Frequency axis.
        temperature : np.ndarray
            Temperature values in Kelvin.
        method : str, optional
            Fit backend. Only 'scipy' is supported for this non-linear model.
            Default: 'scipy'.
        initial_guess : tuple[float, ...], optional
            Initial guess for scipy fit: (a0, a1, b, a2, a3).
            Must have length 5.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'coefficients' (np.ndarray: [a0, a1, b, a2, a3])
            - 'fitted', 'residuals', 'r2', 'covariance', 'valid_mask'
            Also includes convenience keys 'a0', 'a1', 'b', 'a2', 'a3'.
            
        Examples
        --------
        >>> freq = np.array([10e6, 100e6, 300e6])
        >>> temp_k = np.array([80, 90, 100])
        >>> result = LNAModel.fit_temperature(freq, temp_k, method='scipy')
        >>> print(f"R²: {result['r2']:.5f}")
        >>> print(f"a0: {result['a0']:.2e}, b: {result['b']:.2e}")
        
        Notes
        -----
        This is intentionally separate from fit_polynomial_metric to keep the generic
        polynomial fitter simple while supporting this LNA-specific model.
        """
        frequency = np.asarray(frequency, dtype=float)
        temperature = np.asarray(temperature, dtype=float)

        valid = np.isfinite(frequency) & np.isfinite(temperature)
        f = frequency[valid]
        t = temperature[valid]

        # Inverse-frequency term requires non-zero frequencies.
        nonzero = f != 0
        fit_mask = np.zeros_like(valid, dtype=bool)
        fit_mask[np.where(valid)[0][nonzero]] = True
        f = f[nonzero]
        t = t[nonzero]

        min_samples = 5
        if f.size < min_samples:
            raise ValueError(
                f'Frequency points less than {min_samples} after filtering for finite and non-zero values. '
                f'Need at least {min_samples} valid non-zero data points for '
                f'LNA temperature fit.'
            )

        method = method.lower() if method else 'scipy'

        if method != 'scipy':
            raise ValueError("method must be 'scipy' for the non-linear a1/(f+b) model.")

        if initial_guess is None:
            # Seed linear part first, then use a safe positive shift for b.
            slope, intercept = np.polyfit(f, t, deg=1)
            initial_guess = np.array([0.0, 0.0, max(np.median(f), 1.0), intercept, slope], dtype=float)

        initial_guess = np.asarray(initial_guess, dtype=float)
        if initial_guess.size != 5:
            raise ValueError('initial_guess must have length 5: (a0, a1, b, a2, a3).')

        # Keep denominator (f + b) away from zero within fit domain.
        min_f = np.min(f)
        eps = max(1e-12 * max(abs(min_f), 1.0), 1.0)
        lower_bounds = np.array([-np.inf, -np.inf, -min_f + eps, -np.inf, -np.inf], dtype=float)
        upper_bounds = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=float)

        coefficients, covariance = curve_fit(
            LNAModel.model_eval,
            f,
            t,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=50000,
        )

        fitted = LNAModel.model_eval(f, *coefficients)
        residuals = t - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((t - np.mean(t))**2)
        r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

        result = {
            'coefficients': np.asarray(coefficients, dtype=float),
            'fitted': fitted,
            'residuals': residuals,
            'r2': float(r2),
            'covariance': covariance,
            'valid_mask': fit_mask,
        }

        return result
    
    @staticmethod
    def fit_gain(frequency, gain, order=4, method='numpy', initial_guess=None):
        """Fit LNA gain with an Nth-order polynomial.
        
        This is a convenience wrapper around fit_polynomial_metric for gain fitting.

        Parameters
        ----------
        frequency : np.ndarray
            Frequency axis.
        gain : np.ndarray
            Gain values (in dB or linear scale).
        order : int, optional
            Polynomial order (default: 4).
        method : str, optional
            Fit backend: 'numpy' (polyfit) or 'scipy' (curve_fit). Default: 'numpy'.
        initial_guess : tuple[float, ...], optional
            Initial guess for scipy fit: (a0, a1, ..., aN).
            Must have length order + 1.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'coefficients' (np.ndarray: [a0, a1, ..., aN])
            - 'order', 'fitted', 'residuals', 'r2', 'covariance', 'valid_mask'
            Also includes convenience keys 'a0', 'a1', ..., 'aN'.
            
        Examples
        --------
        >>> freq = np.array([10e6, 20e6, 30e6])
        >>> gain_db = np.array([15.2, 15.5, 15.7])
        >>> result = LNAModel.fit_gain(freq, gain_db, order=2, method='numpy')
        >>> print(f"R²: {result['r2']:.5f}")
        >>> print(f"Coefficients: {result['coefficients']}")
        
        Notes
        -----
        Uses fit_polynomial_metric internally. See that function for more details.
        """
        return fit_polynomial_metric(
            frequency, gain, order=order, metric_name='gain',
            method=method, initial_guess=initial_guess
        )
    
    @staticmethod
    def johnson_voltage(T, Z, B=1):
        """Calculate the Johnson-Nyquist noise voltage, in unit of Volts/sqrt(B Hz).
        
        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        Z : complex
            Impedance (ohms).
        B : float, optional
            Bandwidth in Hz (default: 1).
        
        Returns
        -------
        float or np.ndarray
            Johnson voltage in V/sqrt(Hz).
        """
        R = np.real(Z)
        return np.sqrt(4 * k_B * T * B * R)
    
    @staticmethod
    def load_power(V_source, Z_source, Z_load):
        """Calculate the power delivered to a load from a source voltage.
        
        Parameters
        ----------
        V_source : float or np.ndarray
            Source voltage (volts).
        Z_source : complex or np.ndarray
            Source impedance (ohms).
        Z_load : complex or np.ndarray
            Load impedance (ohms).
        
        Returns
        -------
        float or np.ndarray
            Power delivered to load in Watts.
        """
        V_load = V_source * (Z_load / (Z_source + Z_load))
        I = V_load / Z_load
        return np.real(V_load * np.conj(I))
    
    @staticmethod
    def power_delivered_from_s11(source_ntwk, load_ntwk, T_source, Z0=50, B=1):
        """Calculate the power delivered to a load from a Johnson noise source.
        
        Computes power delivered based on source and load reflection coefficients (S11),
        source temperature, and characteristic impedance.

        Parameters
        ----------
        source_ntwk : rf.Network
            Network object with source reflection coefficient (S11).
        load_ntwk : rf.Network
            Network object with load reflection coefficient (S11).
        T_source : float
            Source temperature in Kelvin.
        Z0 : float, optional
            Characteristic impedance (default: 50 ohms).
        B : float, optional
            Bandwidth in Hz (default: 1).

        Returns
        -------
        rf.Network
            Network object with power in Kelvin per B Hz.
        """
        rho_source = source_ntwk.s[:, 0, 0]
        rho_load = load_ntwk.s[:, 0, 0]
        
        impd_source = impedance_from_s11(rho_source, Z0)
        impd_load = impedance_from_s11(rho_load, Z0)

        f = source_ntwk.f

        V_src = LNAModel.johnson_voltage(T_source, impd_source, B)
        P_transferred = LNAModel.load_power(V_src, impd_source, impd_load)
        P_dbm = watt_to_dbm(P_transferred)
        P_kelvin = dbm_to_kelvin(P_dbm)
        power_ntwk = rf.Network(s=P_kelvin, f=f)
        
        return power_ntwk


# ==============================================================================
# Backward compatibility aliases (deprecated)
# ==============================================================================
# These functions are kept for backward compatibility with old code.
# New code should use LNAModel class methods directly.

def johnson_voltage(T, Z, B=1):
    """Deprecated: Use LNAModel.johnson_voltage instead."""
    return LNAModel.johnson_voltage(T, Z, B)


def load_power(V_source, Z_source, Z_load):
    """Deprecated: Use LNAModel.load_power instead."""
    return LNAModel.load_power(V_source, Z_source, Z_load)


def power_delivered_from_s11(source_ntwk, load_ntwk, T_source, Z0=50, B=1):
    """Deprecated: Use LNAModel.power_delivered_from_s11 instead."""
    return LNAModel.power_delivered_from_s11(source_ntwk, load_ntwk, T_source, Z0, B)


def fit_lna_gain(frequency, gain, order=4, method='numpy', initial_guess=None):
    """Deprecated: Use LNAModel.fit_gain instead."""
    return LNAModel.fit_gain(frequency, gain, order, method, initial_guess)


def compute_spike_height_ratios(spike_data1, spike_data2, tolerance=0.1):
    """
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

def apply_gain_to_power(gain_ntwk, input_power, in_offset=0, out_offset=0):
    """Return the output power (minus out_offset) as rf.ntwk object given an input power level and a gain network."""
    ratio_mag = np.abs(gain_ntwk.s[:, 0, 0])
    output_power = (input_power - in_offset) * ratio_mag - out_offset
    output_ntwk = rf.Network(s=output_power.reshape(-1, 1, 1), f=gain_ntwk.f)
    return output_ntwk


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
