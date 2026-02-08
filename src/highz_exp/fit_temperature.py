import numpy as np
import copy
import matplotlib.pyplot as plt
from os.path import join as pjoin
from . import plotter
from scipy.constants import Boltzmann as k_B
from .spec_proc import smooth_spectrum
from .spec_class import Spectrum

class Y_Factor_Thermometer:
    """
    Class to handle Y-Factor temperature measurements and calculations.
    """
    def __init__(self, frequency, DUT_hot, DUT_cold, DUT_name, T_hot, T_cold, cal_hot=None, cal_cold=None, RBW=None):
        """
        Initialize with DUT hot and cold spectra. Both in units of milliwatt.

        Parameters:
            - frequency (np.ndarray): Frequency axis in Hz.
            - DUT_hot (np.ndarray): Measured spectrum with DUT connected at hot source temperature, in mW.
            - DUT_cold (np.ndarray): Measured spectrum with DUT connected at cold source temperature, in mW.
            - frequency (np.ndarray): Frequency axis in Hz.
            - DUT_name (str): Name/label for the DUT.
            - T_hot (float): Hot source temperature in Kelvin.
            - T_cold (float): Cold source temperature in Kelvin.
            - cal_hot (np.ndarray, optional): Calibration spectrum without DUT at hot source temperature, in mW.
            - cal_cold (np.ndarray, optional): Calibration spectrum without DUT at cold source temperature, in mW.
            - RBW (float, optional): Resolution Bandwidth in Hz, required if no calibration spectra provided.
        """
        self.DUT_hot = np.array(DUT_hot)
        self.DUT_cold = np.array(DUT_cold)
        self.Y_factor = self.DUT_hot / self.DUT_cold
        self.CAL_hot = cal_hot
        self.CAL_cold = cal_cold
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.RBW = RBW
        self.label = DUT_name
        self.frequency = frequency
        self.T_sys = self.compute_system_temperature(self.Y_factor, T_hot, T_cold)
        if self.CAL_hot is not None and self.CAL_cold is not None:
            self.g = self.gain_with_cal(self.DUT_hot, self.DUT_cold, self.CAL_hot, self.CAL_cold)
            self.T_cal = self.compute_system_temperature(self.CAL_hot / self.CAL_cold, T_hot, T_cold)
            self.T_dut = self.DUT_temp_with_cal(self.T_cal, self.g, self.T_sys)
        else:
            self.g = None # Gain will be calculated later
            if self.RBW is not None:
                self.g = self.gain_wo_cal(self.DUT_hot*1e-3, self.DUT_cold*1e-3, T_hot, T_cold, self.RBW)
            self.T_dut = None
            self.T_cal = None
    
    @property
    def f(self) -> np.ndarray:
        """Frequency axis in Hz."""
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
            - DUT_hot (np.ndarray): Measured spectrum with DUT connected at hot source temperature, in Watts
            - DUT_cold (np.ndarray): Measured spectrum with DUT connected at cold source temperature, in Watts
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
    
    def smooth_gain(self, kwargs={}):
        """Smooth the gain spectrum using specified smoothing parameters.

        Parameters:
            - kwargs (dict): Keyword arguments for the smoothing function.
        """
        smoothed_gain = smooth_spectrum(self.g, **kwargs)
        return smoothed_gain
    
    def plot_gain(self, **kwargs):
        f_mhz = self.f / 1e6  # Convert frequency to MHz
        plotter.plot_gain(f_mhz, self.g, **kwargs)
    
    def save(self, save_path):
        """
        Save the DUT temperature object to a pickle file.

        Parameters:
            - save_dir (str): Directory to save the file.
            - filename (str): Name of the pickle file.
        """
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    def plot_system_temperature(self, **kwargs):
        """Plot the system temperature spectrum."""
        f_mhz = self.f / 1e6  # Convert frequency to MHz
        plotter.plot_system_temperature(f_mhz, self.T_sys, **kwargs)
    
    @staticmethod
    def plot_temps(faxis: np.ndarray, temp_values: list[np.ndarray], labels, start_freq=10, end_freq=400, ymax=None,
                     title="DUT Temperature", xlabel="Frequency (MHz)", ylabel="Temperature (Kelvin)", save_path=None,
                     marker_freqs=None, smoothing=False, smoothing_kwargs={}):
        """Plot temperature of an component (referred to INPUT of the LNA) curves based on fitted line parameters.

        Parameters:
            - faxis (np.ndarray): Frequency axis in MHz.
            - temp_values (list of np.ndarray): List of temperature arrays at different frequencies.
            - labels (list of str): Labels for each curve.
            - marker_freqs (list of float, optional): Frequencies at which to place vertical markers (in MHz).
            - smoothing (bool, optional): Whether to apply smoothing to the temperature curves.
            - smoothing_kwargs (dict, optional): Additional keyword arguments for the smoothing function.
        
        """

        # Find the index closest to start_freq and end_freq
        start_idx = np.argmin(np.abs(faxis - start_freq))
        end_idx = np.argmin(np.abs(faxis - end_freq))

        plt.figure(figsize=(12, 8))

        # Plot each fitted line
        for temp, label in zip(temp_values, labels):
            plt.plot(faxis[start_idx:end_idx+1], temp[start_idx:end_idx+1], label=label)
            if smoothing:
                smoothed_temp = smooth_spectrum(temp[start_idx:end_idx+1], **smoothing_kwargs)
                plt.plot(faxis[start_idx:end_idx+1], smoothed_temp, linestyle='--',
                         label=f'{label} (smoothed)')

        # Add a vertical marker at the starting frequency
        # plt.axvline(x=faxis[start_idx], color='red', linestyle='--', alpha=0.7,
        #            label=f'Start: {faxis[start_idx]} MHz')
        if ymax is not None:
            plt.ylim(0, ymax)
        
        if marker_freqs is not None:
            for mf in marker_freqs:
            # Find closest index
                idx = np.argmin(np.abs(faxis - mf))
                marker_temp = temp_values[0][idx] if isinstance(temp_values[0], np.ndarray) else temp_values[0]
                marker_freq_mhz = faxis[idx]

                # Plot marker
                plt.plot(marker_freq_mhz, marker_temp, 'ro')
                plt.annotate(f'{marker_temp:.2f} K\n@ {mf:.0f} MHz',
                        (marker_freq_mhz, marker_temp),
                        textcoords="offset points", xytext=(10, 10), ha='left',
                        fontsize=16, color='darkred')

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
    
    def smooth(self, inplace=False, smoothing_kwargs={}):
        """Smooth the gain, system temperature, and DUT temperature spectra."""
        def _smooth_attributes(obj):
            obj.g = smooth_spectrum(obj.g, **smoothing_kwargs)
            obj.T_sys = smooth_spectrum(obj.T_sys, **smoothing_kwargs)
            if obj.T_dut is not None:
                obj.T_dut = smooth_spectrum(obj.T_dut, **smoothing_kwargs)
            return obj
        
        if inplace:
            return _smooth_attributes(self)
        else:
            new_thermo = copy.deepcopy(self)
            return _smooth_attributes(new_thermo)

    def dut_temp_with_known_gain(self) -> np.ndarray:
        pass
    
    def infer_temperature(self, spectrum: Spectrum, start_freq=10, end_freq=400,
                        marker_freqs=None,
                        smoothing='savgol', window_size=31, 
                        ymin=None, ymax=None, title=None, 
                        save_path=None):
        """
        Plot temperature inference of a noise source with optional smoothing.
        This uses system gain and system temperature instead of just the DUT that's being measured. In other words,
        this infers the temperature at the input of the LNA.
        
        Automatically interpolates the gain and system temperature to match the frequency axis of the input spectrum.
        A smoothed curve is plotted alongside the raw inferred temperature for better visualization. 

        Parameters:
            - `spectrum`: Spectrum
                Spectrum object containing frequency in Hz and spectrum data in kelvin.
            - `start_freq` : float, optional
                Start frequency in MHz for plotting.
            - `end_freq` : float, optional
                End frequency in MHz for plotting.
            - `marker_freqs` : list of float, optional
                Frequencies at which to place markers (in MHz).
            - `smoothing` : str, optional
                Type of smoothing: 'savgol' (Savitzky-Golay), 'moving_avg', or 'lowess'
            - `window_size` : int, optional
                Window size for smoothing (must be odd for savgol)
                
        """
        f = spectrum.freq/1e6  # MHz
        spec = spectrum.spec
    
        # Find the index closest to start_freq and end_freq
        start_idx = np.argmin(np.abs(f - start_freq))
        end_idx = np.argmin(np.abs(f - end_freq))

        # conversion of gain from dB to linear scale
        g_values = 10**(self.g / 10)
        noise_values = self.T_sys * g_values

        y_arr = np.asarray(spec, dtype=float)
        
        # interpolate gain and noise temp to match frequency axis of the input spectrum
        g_arr = np.interp(f, self.f/1e6, g_values)
        b_arr = np.interp(f, self.f/1e6, noise_values)

        temp_arr = (y_arr - b_arr)/g_arr

        # Extract the frequency range
        freq_range = f[start_idx:end_idx+1]
        temp_range = temp_arr[start_idx:end_idx+1]

        # Apply smoothing
        smoothed = smooth_spectrum(temp_range, method=smoothing, window=window_size)

        plt.figure(figsize=(12, 8))

        # Plot raw data
        plt.plot(freq_range, temp_range, 'o', alpha=0.4, markersize=6,
                label='Raw data', color='steelblue')

        # Plot smoothed line
        plt.plot(freq_range, smoothed, '-', linewidth=2.5,
                label=f'Smoothed (window={window_size})', color='darkred')
        
        if marker_freqs is not None:
            for mf in marker_freqs:
                # Find closest index
                idx = np.argmin(np.abs(freq_range - mf))
                marker_temp = smoothed[idx]
                marker_freq_mhz = freq_range[idx]

                # Plot marker
                plt.plot(marker_freq_mhz, marker_temp, 'ro')
                plt.annotate(f'{marker_temp:.2f} K\n@ {mf:.0f} MHz',
                             (marker_freq_mhz, marker_temp),
                             textcoords="offset points", xytext=(10, 10), ha='left',
                             fontsize=16, color='darkred')

        if ymax is not None:
            plt.ylim(top=ymax)
        
        if ymin is not None:
            plt.ylim(bottom=ymin)

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
        
        inferred_spectrum = Spectrum(frequency=spectrum.freq, spectrum=temp_arr, name=spectrum.name)

        return inferred_spectrum

