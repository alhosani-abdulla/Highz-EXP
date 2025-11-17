import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from scipy.constants import Boltzmann as k_B

def gain_with_cal(DUT_hot, DUT_cold, cal_hot, cal_cold) -> np.ndarray:
    """
    Calculation of DUT gain with the second-stage/instrument calibration spectra without DUT at two source noise temperatures.
    
    Parameters:
        - DUT_hot (np.ndarray): Measured spectrum with DUT connected at hot source temperature.
        - DUT_cold (np.ndarray): Measured spectrum with DUT connected at cold source temperature.
        - cal_hot (np.ndarray): Calibration spectrum without DUT at hot source temperature.
        - cal_cold (np.ndarray): Calibration spectrum without DUT at cold source temperature.

    Returns:
        - g_dut (np.ndarray): Inferred gain of the DUT at each frequency in db scale.
    """
    g_dut = 10 * np.log10((DUT_hot - DUT_cold) / (cal_hot - cal_cold))
    return g_dut
    
def infer_temperature(faxis, g_values, b_values, y_values, start_freq=10, end_freq=400,
                     smoothing='savgol', window_size=31, title=None, save_path=None):
    """
    Plot temperature inference with optional smoothing.

    Parameters:
    -----------
    smoothing : str, optional
        Type of smoothing: 'savgol' (Savitzky-Golay), 'moving_avg', or 'lowess'
    window_size : int, optional
        Window size for smoothing (must be odd for savgol)
    """
    from scipy.signal import savgol_filter
    from scipy.ndimage import uniform_filter1d

    start_idx = np.argmin(np.abs(faxis - start_freq))
    end_idx = np.argmin(np.abs(faxis - end_freq))

    y_arr = np.asarray(y_values, dtype=float)
    g_arr = np.asarray(g_values, dtype=float)
    b_arr = np.asarray(b_values, dtype=float)
    x_arr = (y_arr - b_arr)/g_arr

    # Extract the frequency range
    freq_range = faxis[start_idx:end_idx+1]
    temp_range = x_arr[start_idx:end_idx+1]

    # Apply smoothing
    if smoothing == 'savgol':
        # Savitzky-Golay filter (preserves peaks better)
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd
        smoothed = savgol_filter(temp_range, window_size, polyorder=3)
    elif smoothing == 'moving_avg':
        # Simple moving average
        smoothed = uniform_filter1d(temp_range, size=window_size, mode='nearest')
    else:
        smoothed = temp_range  # No smoothing

    plt.figure(figsize=(12, 8))

    # Plot raw data
    plt.plot(freq_range, temp_range, 'o', alpha=0.4, markersize=6,
             label='Raw data', color='steelblue')

    # Plot smoothed line
    plt.plot(freq_range, smoothed, '-', linewidth=2.5,
             label=f'Smoothed (window={window_size})', color='darkred')

    plt.xlabel('Frequency (MHz)', fontsize=20)
    plt.ylabel('Temperature (Kelvin)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.title(title, fontsize=22)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return smoothed  # Return smoothed data if needed for further analysis

def plot_hot_cold_gain(faxis, g_values, labels, start_freq=10, end_freq=400, title="Fitted System Gain", xlabel="Frequency", ylabel="Gain (dB)", save_path=None):
    """
    Plot gain(s) of components inferred from hot-cold method in units of dB.
    
    Parameters:
    - faxis (np.ndarray): Frequency axis in MHz.
    - g_values (list of np.ndarray): List of gain arrays, each array corresponds to a component.
    - labels (list of str): Labels for each curve.
    """

    # Find the index closest to start_freq and end_freq
    start_idx = np.argmin(np.abs(faxis - start_freq))
    end_idx = np.argmin(np.abs(faxis - end_freq))

    plt.figure(figsize=(12, 8))

    for g, label in zip(g_values, labels):
        plt.plot(faxis[start_idx:end_idx+1], g[start_idx:end_idx+1], label=label)

    # Add a vertical marker at the starting frequency
    # plt.axvline(x=faxis[start_idx], color='red', linestyle='--', alpha=0.7,
    #            label=f'Start: {faxis[start_idx]} MHz')

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.title(title, fontsize=22)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
      plt.savefig(save_path)
    plt.show()
    
def plot_temps(faxis, g_values, b_values, labels, start_freq=10, end_freq=400,
                     title="Fitted Line Parameters", xlabel="Frequency (MHz)", ylabel="temperature (Kelvin)", save_path=None):
    """
    Plot temperature of an component (referred to INPUT of the LNA) curves based on fitted line parameters.

    Parameters:
    - faxis (np.ndarray): Frequency axis in MHz.
    - g_values (list of np.ndarray): List of gain arrays at different frequencies.
    - b_values (list of np.ndarray): List of noise temperature (referred to OUTPUT) arrays at different frequencies.
    - labels (list of str): Labels for each curve.
    """

    # Find the index closest to start_freq and end_freq
    start_idx = np.argmin(np.abs(faxis - start_freq))
    end_idx = np.argmin(np.abs(faxis - end_freq))

    plt.figure(figsize=(12, 8))

    # Plot each fitted line
    for g, b, label in zip(g_values, b_values, labels):
        plt.plot(faxis[start_idx:end_idx+1], (b/g)[start_idx:end_idx+1], label=label)

    # Add a vertical marker at the starting frequency
    # plt.axvline(x=faxis[start_idx], color='red', linestyle='--', alpha=0.7,
    #            label=f'Start: {faxis[start_idx]} MHz')

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.title(title, fontsize=22)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
      plt.savefig(save_path)
    plt.show()