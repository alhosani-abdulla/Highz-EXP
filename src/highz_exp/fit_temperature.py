import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from scipy.constants import Boltzmann as k_B
from .spec_proc import smooth_spectrum

class Y_Factor_Thermoeter:
    """
    Class to handle Y-Factor temperature measurements and calculations.
    """
    def __init__(self, DUT_hot, DUT_cold, DUT_name, T_hot, T_cold, cal_hot=None, cal_cold=None, RBW=None):
        """
        Initialize with DUT hot and cold spectra. Both in units of milliwatt.

        Parameters:
            - DUT_hot (np.ndarray): Measured spectrum with DUT connected at hot source temperature.
            - DUT_cold (np.ndarray): Measured spectrum with DUT connected at cold source temperature.
            - frequency (np.ndarray): Frequency axis in MHz.
            - DUT_name (str): Name/label for the DUT.
            - T_hot (float): Hot source temperature in Kelvin.
            - T_cold (float): Cold source temperature in Kelvin.
            - cal_hot (np.ndarray, optional): Calibration spectrum without DUT at hot source temperature.
            - cal_cold (np.ndarray, optional): Calibration spectrum without DUT at cold source temperature
            - RBW (float, optional): Resolution Bandwidth in Hz, required if no calibration spectra provided.
        """
        self.DUT_hot = np.array(DUT_hot)
        self.DUT_cold = np.array(DUT_cold)
        self.Y_factor = self.DUT_hot / self.DUT_cold
        self.CAL_hot = cal_hot
        self.CAL_cold = cal_cold
        self.RBW = RBW
        self.label = DUT_name
        self.T_sys = self.compute_system_temperature(self.Y_factor, T_hot, T_cold)
        if self.CAL_hot is not None and self.CAL_cold is not None:
            self.g = self.gain_with_cal(self.DUT_hot, self.DUT_cold, self.CAL_hot, self.CAL_cold)
            self.T_cal = self.compute_system_temperature(self.CAL_hot / self.CAL_cold, T_hot, T_cold)
            self.T_dut = self.DUT_temp_with_cal(self.T_cal, self.g, self.T_sys)
        else:
            self.g = None # Gain will be calculated later
            if self.RBW is not None:
                self.g = self.gain_wo_cal(self.DUT_hot, self.DUT_cold, T_hot, T_cold, self.RBW)
            self.T_dut = None
            self.T_cal = None
    
    @property
    def f(self) -> np.ndarray:
        """Frequency axis in MHz."""
        return self.frequency
    @f.setter
    def f(self, value: np.ndarray) -> None:
        """Set frequency axis."""
        self.frequency = np.array(value)
    
    @property
    def Y(self) -> np.ndarray:
        """Y-Factor array."""
        return self.Y_factor
    @Y.setter
    def Y(self, value: np.ndarray) -> None:
        """Set Y-Factor array."""
        self.Y_factor = np.array(value)
    
    @staticmethod
    def compute_system_temperature(Y_factor, T_hot, T_cold) -> np.ndarray:
        """
        Compute the system temperature based on hot and cold source temperatures.

        Parameters:
            - Y_factor (np.ndarray): Y-Factor array.
            - T_hot (float): Hot source temperature in Kelvin.
            - T_cold (float): Cold source temperature in Kelvin.
        """
        T_sys = (T_hot - Y_factor * T_cold) / (Y_factor - 1)
        return T_sys
    
    @staticmethod
    def gain_wo_cal(DUT_hot, DUT_cold, T_hot, T_cold, RBW) -> np.ndarray:
        """
        Calculation of DUT gain without calibration spectra at two source noise temperatures.

        Parameters:
            - DUT_hot (np.ndarray): Measured spectrum with DUT connected at hot source temperature.
            - DUT_cold (np.ndarray): Measured spectrum with DUT connected at cold source temperature.
            - T_hot (float): Hot source temperature in Kelvin.
            - T_cold (float): Cold source temperature in Kelvin.
            - RBW (float): Resolution Bandwidth in Hz.

        Returns:
            - g_dut (np.ndarray): Inferred gain of the DUT at each frequency in db scale.
        """
        g_dut = 10 * np.log10((DUT_hot - DUT_cold) / ((T_hot - T_cold) * k_B * RBW))
        return g_dut
    
    @staticmethod
    def gain_with_cal(DUT_hot, DUT_cold, cal_hot, cal_cold) -> np.ndarray:
        """
        Calculation of DUT gain with the second-stage/instrument calibration spectra without DUT at two source noise temperatures.
        
        Parameters:
            - cal_hot (np.ndarray): Calibration spectrum without DUT at hot source temperature.
            - cal_cold (np.ndarray): Calibration spectrum without DUT at cold source temperature.

        Returns:
            - g_dut (np.ndarray): Inferred gain of the DUT at each frequency in db scale.
        """
        g_dut = 10 * np.log10((DUT_hot - DUT_cold) / (cal_hot - cal_cold))
        return g_dut
    
    @staticmethod
    def DUT_temp_with_cal(T_cal, g_dut, T_sys) -> np.ndarray:
        """Compute DUT temperature with total system temperature, instrument temperature and DUT gain.
        
        Parameters:
            - T_cal (np.ndarray): Instrument temperature spectrum without DUT at each frequency in Kelvin.
            - g_dut (np.ndarray): Inferred gain of the DUT at each frequency in db scale.
            - T_sys (np.ndarray): Total system temperature spectrum with DUT at each frequency in Kelvin
        """
        T_dut = T_sys - T_cal / (10**(g_dut / 10))
        return T_dut
    
    @staticmethod
    def plot_temps(faxis, temp_values, labels, start_freq=10, end_freq=400, ymax=None,
                     title="DUT Temperature", xlabel="Frequency (MHz)", ylabel="temperature (Kelvin)", save_path=None):
        """
        Plot temperature of an component (referred to INPUT of the LNA) curves based on fitted line parameters.

        Parameters:
        - faxis (np.ndarray): Frequency axis in MHz.
        - temp_values (list of np.ndarray): List of temperature arrays at different frequencies.
        - labels (list of str): Labels for each curve.
        """

        # Find the index closest to start_freq and end_freq
        start_idx = np.argmin(np.abs(faxis - start_freq))
        end_idx = np.argmin(np.abs(faxis - end_freq))

        plt.figure(figsize=(12, 8))

        # Plot each fitted line
        for temp, label in zip(temp_values, labels):
            plt.plot(faxis[start_idx:end_idx+1], temp[start_idx:end_idx+1], label=label)

        # Add a vertical marker at the starting frequency
        # plt.axvline(x=faxis[start_idx], color='red', linestyle='--', alpha=0.7,
        #            label=f'Start: {faxis[start_idx]} MHz')
        if ymax is not None:
            plt.ylim(0, ymax)

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
    # Find the index closest to start_freq and end_freq
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
    smoothed = smooth_spectrum(temp_range, method=smoothing, window=window_size)

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

