import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle
import skrf as rf
from os.path import join as pjoin
from . import plotter
from scipy.constants import Boltzmann as k_B
from . import spec_proc, unit_convert
from .spec_class import Spectrum
from scipy.signal import medfilt

class Y_Factor_Thermometer:
    """
    Class to handle Y-Factor temperature measurements and calculations.
    """
    def __init__(self, frequency, DUT_hot, DUT_cold, DUT_name, T_hot, T_cold, 
                 cal_hot=None, cal_cold=None, RBW=None, unit=None):
        """
        Initialize with DUT hot and cold spectra. Both in units of milliwatt.

        Parameters:
            - frequency (np.ndarray): Frequency axis in Hz.
            - DUT_hot (np.ndarray): Measured spectrum with DUT connected at hot source temperature, in linear power units.
            - DUT_cold (np.ndarray): Measured spectrum with DUT connected at cold source temperature, in linear power units.
            - frequency (np.ndarray): Frequency axis in Hz.
            - DUT_name (str): Name/label for the DUT.
            - T_hot (float): Hot source temperature in Kelvin.
            - T_cold (float): Cold source temperature in Kelvin.
            - cal_hot (np.ndarray, optional): Calibration spectrum without DUT at hot source temperature, in linear power units.
            - cal_cold (np.ndarray, optional): Calibration spectrum without DUT at cold source temperature, in linear power units.
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
        self.unit = unit
        self.frequency = frequency
        self.T_sys = self.compute_system_temperature(self.Y_factor, T_hot, T_cold)
        self.g = None # Gain will be calculated later
        self.g_sys = None

        if self.CAL_hot is not None and self.CAL_cold is not None:
            self.g = self.gain_with_cal(self.DUT_hot, self.DUT_cold, self.CAL_hot, self.CAL_cold)
            self.T_cal = self.compute_system_temperature(self.CAL_hot / self.CAL_cold, T_hot, T_cold)
            self.T_dut = self.DUT_temp_with_cal(self.T_cal, self.g, self.T_sys)
            if self.RBW is not None: self.g_sys = self.gain_wo_cal(self.DUT_hot, self.DUT_cold, T_hot, T_cold, self.RBW, self.unit)
        else:
            if self.RBW is not None: self.g = self.gain_wo_cal(self.DUT_hot, self.DUT_cold, T_hot, T_cold, self.RBW, self.unit)
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
    def gain_wo_cal(DUT_hot, DUT_cold, T_hot, T_cold, RBW, unit) -> np.ndarray:
        """
        Calculation of DUT gain without calibration spectra at two source noise temperatures.

        Parameters:
            - DUT_hot (np.ndarray): Measured spectrum with DUT connected at hot source temperature
            - DUT_cold (np.ndarray): Measured spectrum with DUT connected at cold source temperature
            - T_hot (float): Hot source temperature in Kelvin.
            - T_cold (float): Cold source temperature in Kelvin.
            - RBW (float): Resolution Bandwidth in Hz.
            - unit (str): Unit of the input spectra (DUT_hot, DUT_cold), either 'dBm', 'Kelvin', or None for linear power in Watts.

        Returns:
            - g_dut (np.ndarray): Inferred gain of the DUT at each frequency in db scale.
        """
        if unit is None:
            g = 10 * np.log10((DUT_hot - DUT_cold) / ((T_hot - T_cold)))
        elif unit.lower() == 'dbm':
            DUT_hot = unit_convert.dbm_to_milliwatt(DUT_hot) * 1e-3
            DUT_cold = unit_convert.dbm_to_milliwatt(DUT_cold) * 1e-3
            g = 10 * np.log10((DUT_hot - DUT_cold) / ((T_hot - T_cold) * k_B * RBW))
        elif unit.lower() == 'kelvin':
            g = 10 * np.log10((DUT_hot - DUT_cold) / ((T_hot - T_cold)))
        elif unit.lower() == 'milliwatt':
            g = 10 * np.log10(((DUT_hot - DUT_cold)) * 1e-3 / ((T_hot - T_cold) * k_B * RBW))
        else:
            raise ValueError(f"Unsupported unit '{unit}' for gain calculation. Supported units are 'dBm', 'Kelvin', or None for linear power.")
        return g
    
    def export_gain(self, export_path=None) -> rf.Network:
        """Export the inferred gain spectrum as a scikit-rf Network object.

        Parameters:
            - export_path (str, optional): 
                If provided, the Network object will be saved to this path in Touchstone format.
        """
        ntwk = rf.Network()
        ntwk.f = self.f
        ntwk.s = np.zeros((len(self.f), 2, 2), dtype=complex)  # Initialize S-parameter array
        ntwk.s[:, 1, 0] = 10**(self.g / 20)  # Convert gain from dB to linear scale in voltage gain
        
        if export_path is not None:
            ntwk.write_touchstone(export_path)
        
        return ntwk
    
    def copy(self):
        """Create a copy of the Y_Factor_Thermometer object."""
        return copy.deepcopy(self)
    
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
    
    def smooth_gain(self, inplace=False, **kwargs):
        """Smooth the gain spectrum using specified smoothing parameters.

        Parameters:
            - kwargs (dict): Keyword arguments for the smoothing function.
        """
        if self.g is None:
            raise ValueError("Gain spectrum is not available to smooth. Please check if calibration data and RBW are provided to compute it.")
        if np.any(np.isnan(self.g)):
            self.frequency = self.frequency[~np.isnan(self.g)]
            self.g = self.g[~np.isnan(self.g)]
        smoothed_gain = spec_proc.smooth_spectrum(self.g, **kwargs)
        if inplace:
            self.g = smoothed_gain
        return smoothed_gain
    
    def plot_gain(self, **kwargs) -> Spectrum:
        f_mhz = self.f / 1e6  # Convert frequency to MHz
        plotter.plot_gain(f_mhz, self.g, **kwargs)
        gain_spec = Spectrum(frequency=self.f, spectrum=self.g, name=f'{self.label} Gain')
        return gain_spec

    def plot_sys_gain(self, **kwargs):
        if not hasattr(self, 'g_sys'):
            raise ValueError("System gain spectrum is not available. Please check if all calibration data and RBW are provided to compute it.")
        f_mhz = self.f / 1e6  # Convert frequency to MHz
        plotter.plot_gain(f_mhz, self.g_sys, **kwargs)

    def export_temperature(self, save_dir=None) -> tuple[Spectrum, Spectrum | None]:
        """Export the system temperature and DUT temperature spectra as Spectrum objects."""
        system_temp_spec = Spectrum(frequency=self.f, spectrum=self.T_sys, name=f'{self.label} System Temperature')
        dut_temp_spec = None
        if self.T_dut is not None:
            dut_temp_spec = Spectrum(frequency=self.f, spectrum=self.T_dut, name=f'{self.label} DUT Temperature')
        if save_dir is not None:
            with open(pjoin(save_dir, f'{self.label}_Tsys.pkl'), 'wb') as f:
                pickle.dump(system_temp_spec, f)
            if dut_temp_spec is not None:
                with open(pjoin(save_dir, f'{self.label}_Tdut.pkl'), 'wb') as f:
                    pickle.dump(dut_temp_spec, f)
        return system_temp_spec, dut_temp_spec
    
    def resample(self, new_freq, reducer=np.nanmean, inplace=True,
                 return_uncertainty=False):
        """
        Resample spectra onto a lower-resolution frequency axis by bin-averaging.

        Parameters
        ----------
        new_freq : np.ndarray
            Target frequency bin centers (Hz).
        reducer : callable, optional
            Function used to combine samples inside each bin
            (e.g. np.nanmean, np.nanmedian).
        return_uncertainty : bool, optional
            If True, also return per-bin uncertainty estimates and sample counts.
        inplace : bool, optional
            Modify object in place or return a copy.

        Returns
        -------
        Y_Factor_Thermometer or tuple[Y_Factor_Thermometer, dict]
        """
        new_freq = np.asarray(new_freq)
        edges = _bin_edges_from_centers(new_freq)

        if return_uncertainty:
            T_sys_new, T_sys_unc, _ = _bin_average_with_uncertainty(
                self.frequency, self.T_sys, edges, reducer=reducer
            )
        else:
            T_sys_new = spec_proc._bin_average(self.frequency, self.T_sys, edges, reducer)
            T_sys_unc = None

        T_dut_new = None
        T_dut_unc = None
        if self.T_dut is not None:
            if return_uncertainty:
                T_dut_new, T_dut_unc, _ = _bin_average_with_uncertainty(
                    self.frequency, self.T_dut, edges, reducer=reducer
                )
            else:
                T_dut_new = spec_proc._bin_average(self.frequency, self.T_dut, edges, reducer)

        g_new = None
        g_unc = None
        if self.g is not None:
            if return_uncertainty:
                g_new, g_unc, _ = _bin_average_with_uncertainty(
                    self.frequency, self.g, edges, reducer=reducer
                )
            else:
                g_new = spec_proc._bin_average(self.frequency, self.g, edges, reducer)
        
        if hasattr(self, 'g_sys') and self.g_sys is not None:
            g_sys_new = spec_proc._bin_average(self.frequency, self.g_sys, edges, reducer)
        else:
            g_sys_new = None

        target = self if inplace else copy.deepcopy(self)

        target.frequency = new_freq
        target.T_sys = T_sys_new
        target.T_dut = T_dut_new
        target.g = g_new
        target.T_sys_unc = T_sys_unc
        target.T_dut_unc = T_dut_unc
        target.g_unc = g_unc
        target.g_sys = g_sys_new if hasattr(self, 'g_sys') else None

        return target
    
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
        # convert system temperature to spectrum object
        system_spec = Spectrum(frequency=self.f, spectrum=self.T_sys, name=f'{self.label} System Temperature')
        plotter.plot_spectra([system_spec], ylabel='Temperature (K)', **kwargs)
    
    def plot_dut_temperature(self, **kwargs) -> Spectrum:
        """Plot the DUT temperature spectrum."""
        if self.T_dut is None:
            raise ValueError("DUT temperature spectrum is not available. Please compute it first.")
        dut_spec = Spectrum(frequency=self.f, spectrum=self.T_dut, name=f'{self.label} DUT Temperature')
        plotter.plot_spectra([dut_spec], ylabel='Temperature (K)', **kwargs)
        return dut_spec
    
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
                smoothed_temp = spec_proc.smooth_spectrum(temp[start_idx:end_idx+1], **smoothing_kwargs)
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
            obj.g = spec_proc.smooth_spectrum(obj.g, **smoothing_kwargs)
            obj.T_sys = spec_proc.smooth_spectrum(obj.T_sys, **smoothing_kwargs)
            if obj.T_dut is not None:
                obj.T_dut = spec_proc.smooth_spectrum(obj.T_dut, **smoothing_kwargs)
            return obj
        
        if inplace:
            return _smooth_attributes(self)
        else:
            new_thermo = copy.deepcopy(self)
            return _smooth_attributes(new_thermo)
    
    def medfilt_all(self, kernel_size=31, inplace=False):
        """Apply median filtering to gain, system temperature, and DUT temperature spectra."""
        def _medfilt_attributes(obj):
            obj.g = medfilt(obj.g, kernel_size=kernel_size)
            obj.T_sys = medfilt(obj.T_sys, kernel_size=kernel_size)
            if obj.T_dut is not None:
                obj.T_dut = medfilt(obj.T_dut, kernel_size=kernel_size)
            if obj.g_sys is not None:
                obj.g_sys = medfilt(obj.g_sys, kernel_size=kernel_size)
            return obj
        if inplace:
            return _medfilt_attributes(self)
        else:
            new_thermo = copy.deepcopy(self)
            return _medfilt_attributes(new_thermo)

    def infer_temp_with_known_gain(self, spectrum, s21_ntwk) -> np.ndarray:
        """Calculate the DUT temperature spectrum using a known gain spectrum from an S-parameter measurement.

        Parameters:
            - s21_ntwk (scikit-rf Network): Network object containing the S21 parameter (gain) of the DUT.

        Returns:
            - T_dut (np.ndarray): Inferred DUT temperature spectrum in Kelvin.
        """
        # Extract S21 parameter (gain) from the Network object and convert to linear scale in power gain
        g_s21 = np.abs(s21_ntwk.s[:, 1, 0])**2  
        
        # get three arrays of frequencies
        freq_g = s21_ntwk.f / 1e6  # Convert frequency to MHz
        freq_noise = self.f / 1e6  
        freq_spec = spectrum.freq / 1e6 
        
        # interpolate all arrays to the same frequency axis (that of the input spectrum)
        g_interp = np.interp(freq_spec, freq_g, g_s21)
        noise_interp = np.interp(freq_spec, freq_noise, self.T_sys)
        
        temp_arr = (spectrum.spec - noise_interp) / g_interp

        inferred_spectrum = Spectrum(frequency=spectrum.freq, spectrum=temp_arr, name=spectrum.name)
    
        return inferred_spectrum
    
    def infer_temperature(self, spectrum: Spectrum, freq_range=(None, None),
        marker_freqs=None,
        smoothing='savgol', window_size=31, 
        y_range=(None, None), title=None, show_plot=True,
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
            - `freq_range` : tuple of (float, float), optional
                Frequency range to plot/calculate (fmin, fmax) in MHz.
            - `marker_freqs` : list of float, optional
                Frequencies at which to place markers (in MHz).
            - `smoothing` : str, optional
                Type of smoothing: 'savgol' (Savitzky-Golay), 'moving_avg', or 'lowess'.
                If None, no smoothing is applied.
            - `window_size` : int, optional
                Window size for smoothing (must be odd for savgol)
                
        """
        f = spectrum.freq/1e6  # MHz
        spec = spectrum.spec
    
        # Find the index closest to start_freq and end_freq
        start_freq, end_freq = freq_range
        if start_freq is not None:
            start_idx = np.argmin(np.abs(self.f/1e6 - start_freq))
        else:
            start_idx = 0
        if end_freq is not None:
            end_idx = np.argmin(np.abs(self.f/1e6 - end_freq))
        else:
            end_idx = len(self.f/1e6) - 1

        # conversion of gain from dB to linear scale
        if hasattr(self, 'g_sys') and self.g_sys is not None:
            print("Using system gain for temperature inference.")
            g_values = 10**(self.g_sys / 10)
        else:
            g_values = 10**(self.g / 10)
        noise_values = self.T_sys * g_values

        y_arr = np.interp(self.f/1e6, f, spec)

        temp_arr = (y_arr - noise_values)/g_values

        # Extract the frequency range
        freq_range = (self.f/1e6)[start_idx:end_idx+1]
        temp_range = temp_arr[start_idx:end_idx+1]

        # Apply smoothing
        if smoothing is not None:
            smoothed = spec_proc.smooth_spectrum(temp_range, method=smoothing, window=window_size)

        if show_plot:
            plt.figure(figsize=(12, 8))

            # Plot raw data
            plt.plot(freq_range, temp_range, 'o', alpha=0.4, markersize=6,
                    label='Raw data', color='steelblue')

            # Plot smoothed line if smoothing is applied
            if smoothing is not None:
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

            ymin, ymax = y_range

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
        
        freq_range_hz = freq_range * 1e6
        inferred_spectrum = Spectrum(frequency=freq_range_hz, spectrum=temp_range, name=spectrum.name)

        return inferred_spectrum
    
def _bin_edges_from_centers(f_centers):
    edges = np.zeros(len(f_centers) + 1)
    edges[1:-1] = 0.5 * (f_centers[1:] + f_centers[:-1])
    edges[0] = f_centers[0] - (edges[1] - f_centers[0])
    edges[-1] = f_centers[-1] + (f_centers[-1] - edges[-2])
    return edges

def _bin_average_with_uncertainty(x, y, edges, reducer=np.nanmean):
    """
    Bin-average values and estimate uncertainty per output bin as SEM from in-bin scatter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        mean, sigma, n_samples for each output bin.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    n_bins = len(edges) - 1
    mean_out = np.full(n_bins, np.nan)
    sigma_out = np.full(n_bins, np.nan)
    n_samples_out = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (x >= edges[i]) & (x < edges[i+1])
        if not np.any(mask):
            continue

        yi = np.asarray(y[mask], dtype=float)

        valid = np.isfinite(yi)
        yi = yi[valid]
        if yi.size == 0:
            continue

        mu = reducer(yi)
        mean_out[i] = mu
        n_samples_out[i] = yi.size

        if yi.size > 1:
            sigma_out[i] = np.nanstd(yi, ddof=1) / np.sqrt(yi.size)

    return mean_out, sigma_out, n_samples_out

