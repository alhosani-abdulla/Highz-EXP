from typing import Any, Dict, Iterable, Optional
import numpy as np
from scipy.signal import savgol_filter
from . import spec_proc

"""
spec_class.py

A small utility class for loading and processing spectra.

Usage:
    s = Spectrum(frequency, spectrum, name="recording1", metadata={"laser": "532nm"})
    s.normalize("area").resample(np.linspace(400, 800, 1000)).smooth(51)
"""
class Spectrum:
    """
    Lightweight spectrum container with common processing utilities.

    Args:
        frequency: 1D array-like of frequency (or wavelength) values.
        spectrum: 1D array-like of measured intensities (same length as frequency).
        name: descriptive name for this recorded spectrum.
        metadata: optional dict of additional metadata.

    Notes:
        Methods generally return self to allow method chaining.
    """

    def __init__(
        self,
        frequency: Iterable[float],
        spectrum: Iterable[float],
        name: str,
        colorcode: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.freq = np.asarray(frequency, dtype=float).ravel()
        self.spec = np.asarray(spectrum, dtype=float).ravel()
        if self.freq.shape != self.spec.shape:
            raise ValueError("frequency and spectrum must have the same shape")
        self.name = str(name)
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}
        self.colorcode = colorcode

    @property
    def s(self) -> np.ndarray:
        """Short alias for the spectrum array (read access)."""
        return self.spec

    @s.setter
    def s(self, value: Iterable[float]) -> None:
        """Short alias for the spectrum array (write access)."""
        self.spec = np.asarray(value, dtype=float).ravel()

    @property
    def f(self) -> np.ndarray:
        """Short alias for the frequency array (read access)."""
        return self.freq

    @f.setter
    def f(self, value: Iterable[float]) -> None:
        """Short alias for the frequency array (write access)."""
        self.freq = np.asarray(value, dtype=float).ravel()

    def copy(self) -> "Spectrum":
        """Return a deep copy of the Spectrum."""
        return Spectrum(self.freq.copy(), self.spec.copy(), self.name, self.colorcode, self.metadata)
    
    def plot(self, **plot_kwargs) -> Any:
        from highz_exp.plotter import plot_spectrum
        return plot_spectrum(self, **plot_kwargs)

    def unit_convert(self, from_unit: str, to_unit: str, channel_width: Optional[float] = None,
                     inplace: bool = False) -> "Spectrum":
        """
        Convert spectrum units between 'dBm', 'milliwatt', 'watt', and 'kelvin'.

        Args:
            from_unit: current unit of the spectrum ('dBm', 'milliwatt', 'watt', 'kelvin').
            to_unit: desired unit of the spectrum ('dBm', 'milliwatt', 'watt', 'kelvin').
            channel_width: required when converting to/from 'kelvin' (in Hz).
        Returns:
            Spectrum: self with converted spectrum.
        """
        from highz_exp.unit_convert import dbm_to_milliwatt, watt_to_dbm, dbm_to_kelvin, kelvin_to_dbm
        unit_options = ['dBm', 'milliwatt', 'watt', 'kelvin']
        if from_unit not in unit_options:
            raise ValueError(f"from_unit must be one of {unit_options}")
        if to_unit not in unit_options:
            raise ValueError(f"to_unit must be one of {unit_options}")
        if (from_unit == 'kelvin' or to_unit == 'kelvin') and channel_width is None:
            raise ValueError("channel_width must be provided when converting to/from 'kelvin'")
        spec_converted = self.spec.copy()
        
        # Convert to dBm first
        if from_unit == 'milliwatt':
            spec_converted = watt_to_dbm(spec_converted * 1e-3)
        elif from_unit == 'watt':
            spec_converted = watt_to_dbm(spec_converted)
        elif from_unit == 'kelvin':
            spec_converted = kelvin_to_dbm(spec_converted, channel_width=channel_width)
        # Now convert from dBm to target unit
        if to_unit == 'milliwatt':
            spec_converted = dbm_to_milliwatt(spec_converted)
        elif to_unit == 'watt':
            spec_converted = dbm_to_milliwatt(spec_converted) * 1e-3
        elif to_unit == 'kelvin':
            spec_converted = dbm_to_kelvin(spec_converted, channel_width=channel_width)

        if inplace:
            self.spec = spec_converted
            self.metadata['unit'] = to_unit
            return self
        else:
            return Spectrum(self.freq.copy(), spec_converted, self.name, colorcode=self.colorcode, metadata=self.metadata.copy())

    def resample(self, new_freq: Iterable[float], kind: str = "linear") -> "Spectrum":
        """
        Resample spectrum onto new_freq. Uses numpy.interp for linear interpolation.
        kind currently supports only 'linear'.
        """
        new_freq_arr = np.asarray(new_freq, dtype=float)
        if kind != "linear":
            raise NotImplementedError("only 'linear' interpolation is implemented")
        # ensure monotonic x for interpolation
        if not np.all(np.diff(self.freq) >= 0):
            idx = np.argsort(self.freq)
            freq_sorted = self.freq[idx]
            spec_sorted = self.spec[idx]
        else:
            freq_sorted = self.freq
            spec_sorted = self.spec
        new_spec = np.interp(new_freq_arr, freq_sorted, spec_sorted, left=np.nan, right=np.nan)
        self.freq = new_freq_arr
        self.spec = new_spec
        return self
    
    def despike(self, window: int = 11, threshold: float = 5.0, replace: str = "median") -> "Spectrum":
        """
        Remove narrow RFI spikes by comparing each point to a local median and MAD.

        Parameters:
            window: odd integer window size for local statistics (>=3).
            threshold: multiple of local MAD above which a point is considered a spike.
            replace: 'median' to replace spikes with local median, 'interp' to interpolate
                     across spike points using neighboring good points.

        Notes:
            This uses numpy's sliding_window_view when available, or scipy.signal.medfilt
            as a fallback. Both scipy.signal.medfilt and numpy.lib.stride_tricks.sliding_window_view
            can be used to speed up the local-median computation.
        """
        self.spec = spec_proc.despike(self.spec, window=window, threshold=threshold, replace=replace)

    def smooth(self, window: int = 11, method: str = "savgol", polyorder: int = 3, 
               freq_interval: Optional[float] = None, inplace: bool = False) -> "Spectrum":
        """
        Smooth the spectrum.

        Parameters:
            window: window length in samples (used if freq_interval is None).
            method: 'savgol' (Savitzky-Golay) or 'moving' (simple moving average).
            polyorder: polynomial order for savgol filter.
            freq_interval: optional frequency interval; if provided, window is computed from it.
            inplace: if True, modify the spectrum in place; otherwise, return a new Spectrum object.

        Notes:
            window must be odd for savgol. If freq_interval is provided, it takes precedence.
        """
        # Compute window from freq_interval if provided
        if freq_interval is not None:
            if self.freq.size < 2:
                return self
            df = float(np.abs(self.freq[1] - self.freq[0]))
            window = max(3, int(np.round(freq_interval / df)))
        
        if window < 3:
            return self
        if method == "savgol":
            if savgol_filter is None:
                # fallback to moving average if scipy not available
                method = "moving"
            else:
                wl = window if window % 2 == 1 else window + 1
                wl = max(3, wl)
                try:
                    new_spec = savgol_filter(self.spec, wl, polyorder, mode="interp")
                except Exception:
                    # fallback
                    method = "moving"
        if method == "moving":
            k = int(window)
            if k % 2 == 0:
                k += 1
            pad = k // 2
            padded = np.pad(self.spec, pad, mode="edge")
            kernel = np.ones(k) / k
            new_spec = np.convolve(padded, kernel, mode="valid")

        if inplace:
            self.spec = new_spec
            return self
        else:
            return Spectrum(self.freq.copy(), new_spec, self.name, self.colorcode, self.metadata)     

    def trim(self, min_freq: Optional[float] = None, max_freq: Optional[float] = None) -> "Spectrum":
        """Keep only data between min_freq and max_freq (inclusive)."""
        mask = np.ones_like(self.freq, dtype=bool)
        if min_freq is not None:
            mask &= self.freq >= float(min_freq)
        if max_freq is not None:
            mask &= self.freq <= float(max_freq)
        self.freq = self.freq[mask]
        self.spec = self.spec[mask]
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict of the spectrum and metadata."""
        return {
            "name": self.name,
            "frequency": self.freq.tolist(),
            "spectrum": self.spec.tolist(),
            "metadata": dict(self.metadata),
        }

    def save_npz(self, path: str) -> None:
        """Save spectrum to a .npz file."""
        np.savez_compressed(path, freq=self.freq, spec=self.spec, name=self.name, metadata=self.metadata)

    @classmethod
    def load_npz(cls, path: str) -> "Spectrum":
        """Load from .npz file created by save_npz."""
        d = np.load(path, allow_pickle=True)
        metadata = d.get("metadata", {})
        name = str(d.get("name", "loaded"))
        return cls(d["freq"], d["spec"], name=name, metadata=metadata.tolist() if hasattr(metadata, "tolist") else dict(metadata))

    @classmethod
    def from_two_column_txt(cls, path: str, name: Optional[str] = None, delimiter=None, skiprows: int = 0) -> "Spectrum":
        """
        Load two-column text or CSV: first column frequency, second column spectrum.
        """
        data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("expect file with at least two columns")
        freq = data[:, 0]
        spec = data[:, 1]
        return cls(freq, spec, name=name or path)

    def __repr__(self) -> str:
        return f"<Spectrum name={self.name!r} points={self.freq.size} metadata_keys={list(self.metadata.keys())}>"
    
    def preprocess(self, remove_spikes=True, unit='dBm', offset=-135, system_gain=100, normalize=None) -> "Spectrum":
        """Preprocess the spectrum by converting to the specified unit and removing spikes if required. 
        
        Parameters:
            system_gain: float, the system gain in dB to be discounted from the recorded spectrum.
        Returns:
            Spectrum: The processed Spectrum object.
        """
        import copy
        from .unit_convert import rfsoc_spec_to_dbm, dbm_to_kelvin
        from .spec_proc import remove_spikes_from_psd
        
        copy_spec = copy.deepcopy(self.spec)
        faxis = self.freq
        spectrum = copy_spec

        if remove_spikes:
            spectrum = remove_spikes_from_psd(faxis, spectrum)

        spectrum_dBm = rfsoc_spec_to_dbm(spectrum, offset=offset) - system_gain

        if unit == 'dBm':
            copy_spec.spec = spectrum_dBm
        elif unit == 'kelvin':
            df = float(faxis[1] - faxis[0])
            spectrum_kelvin = dbm_to_kelvin(spectrum_dBm, df)
            if normalize is not None:
                copy_spec.spec =  spectrum_kelvin * normalize
            else:
                copy_spec.spec = spectrum_kelvin
        else:
            raise ValueError("unit must be dBm or kelvin.")
        
        return copy_spec
    
    @staticmethod
    def preprocess_states(load_states, remove_spikes=True, unit='dBm', offset=-135, system_gain=100, normalize=None) -> dict:
        """Preprocess the loaded states by converting the spectrum to the specified unit and removing spikes if required. 
        
        Parameters:
            load_states: dict of Spectrum objects. {state_name: Spectrum, ...}
            system_gain: float, the system gain in dB to be discounted from the recorded spectrum.

        Returns:
            dict: A dictionary of processed network objects. {state_name: Spectrum (processed), ...}
        """
        import copy
        import skrf as rf
        from .unit_convert import rfsoc_spec_to_dbm, dbm_to_kelvin
        from .spec_proc import remove_spikes_from_psd
        
        assert type(load_states) is dict, "load_states must be a dictionary of Spectrum objects."
        for state in load_states.values():
            assert isinstance(state, Spectrum), "All values in load_states must be Spectrum objects."
        
        freq = list(load_states.values())[0].freq

        df = float(freq[1] - freq[0])
        loaded_states_copy = copy.deepcopy(load_states)
        state_dict = {}
        for label, state in loaded_states_copy.items():
            if remove_spikes:
                spectrum = remove_spikes_from_psd(freq, state.spec)
            else: spectrum = state.spec

            spectrum_dBm = rfsoc_spec_to_dbm(spectrum, offset=offset) - system_gain

            if unit == 'dBm':
                state.spec = spectrum_dBm
            elif unit == 'kelvin':
                spectrum = dbm_to_kelvin(spectrum_dBm, df)
                if normalize is not None:
                    state.spec =  spectrum * normalize
                else:
                    state.spec = spectrum
            else:
                raise ValueError("unit must be dBm or kelvin.")
        
        for label, state in loaded_states_copy.items():
            spectrum = state.spec
            state_dict[label] = Spectrum(state.freq, spectrum, name=state.name, metadata=state.metadata,
                                         colorcode=state.colorcode)
        return state_dict

    @staticmethod
    def norm_states(loaded_states, ref_state_label, ref_temp=300, system_gain=100) -> tuple:
        """Normalize loaded RAW! spectra from digital spectrometer to a reference state and convert to Kelvin.

        Returns:
        loaded_states_kelvin: dict
            Dictionary of loaded Spectrum objects with spectra in Kelvin.
        gain: np.ndarray
            Normalization factor applied to convert from dBm to Kelvin.
        """
        from .unit_convert import rfsoc_spec_to_dbm, norm_factor
        from .spec_proc import remove_spikes_from_psd

        freq = loaded_states[list(loaded_states.keys())[0]].freq
        dbm = np.array(rfsoc_spec_to_dbm(remove_spikes_from_psd(freq, loaded_states[ref_state_label].spec)))-system_gain
        gain = norm_factor(dbm, ref_temp)
        loaded_states_kelvin = Spectrum.preprocess_states(loaded_states, unit='kelvin', normalize=gain, system_gain=system_gain)

        return loaded_states_kelvin, gain