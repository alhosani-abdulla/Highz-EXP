from typing import Any, Dict, Iterable, Optional
import numpy as np
from scipy.signal import savgol_filter

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.freq = np.asarray(frequency, dtype=float).ravel()
        self.spec = np.asarray(spectrum, dtype=float).ravel()
        if self.freq.shape != self.spec.shape:
            raise ValueError("frequency and spectrum must have the same shape")
        self.name = str(name)
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}

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
        return Spectrum(self.freq.copy(), self.spec.copy(), self.name, dict(self.metadata))


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

    def smooth(self, window_len: int = 11, method: str = "savgol", polyorder: int = 3) -> "Spectrum":
        """
        Smooth the spectrum.

        method: 'savgol' (Savitzky-Golay) or 'moving' (simple moving average).
        window_len must be odd for savgol.
        """
        if window_len < 3:
            return self
        if method == "savgol":
            if savgol_filter is None:
                # fallback to moving average if scipy not available
                method = "moving"
            else:
                wl = window_len if window_len % 2 == 1 else window_len + 1
                wl = max(3, wl)
                try:
                    self.spec = savgol_filter(self.spec, wl, polyorder, mode="interp")
                except Exception:
                    # fallback
                    method = "moving"
        if method == "moving":
            k = int(window_len)
            if k % 2 == 0:
                k += 1
            pad = k // 2
            padded = np.pad(self.spec, pad, mode="edge")
            kernel = np.ones(k) / k
            self.spec = np.convolve(padded, kernel, mode="valid")
        return self

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