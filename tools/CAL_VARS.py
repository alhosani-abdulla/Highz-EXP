from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class LinearNoiseDiodeProfile:
	"""Two-point linear frequency-to-temperature profile for a noise diode."""

	name: str
	f1_mhz: float
	t1_k: float
	f2_mhz: float
	t2_k: float

	def temperatures_k(self, frequencies_mhz: np.ndarray | float) -> np.ndarray:
		"""Return mapped temperatures in Kelvin for one or many frequencies in MHz."""
		freqs = np.asarray(frequencies_mhz, dtype=float)
		if self.f1_mhz == self.f2_mhz:
			raise ValueError("Profile anchor frequencies must be different.")

		slope_k_per_mhz = (self.t2_k - self.t1_k) / (self.f2_mhz - self.f1_mhz)
		return self.t1_k + slope_k_per_mhz * (freqs - self.f1_mhz)

# ND01 calibration: 50 MHz -> 1976 K, 200 MHz -> 2110 K
ND01 = LinearNoiseDiodeProfile(
	name="ND01",
	f1_mhz=50.0,
	t1_k=1976.0,
	f2_mhz=200.0,
	t2_k=2110.0,
)

# ND02 calibration: 50 MHz -> 2037 K, 200 MHz -> 2177 K
ND02 = LinearNoiseDiodeProfile(
	name="ND02",
	f1_mhz=50.0,
	t1_k=2037.0,
	f2_mhz=200.0,
	t2_k=2177.0,
)


def nd01_temperature_k(frequencies_mhz: np.ndarray | float) -> np.ndarray:
	"""Convenience wrapper for ND01 frequency-to-temperature mapping."""
	return ND01.temperatures_k(frequencies_mhz)

def nd02_temperature_k(frequencies_mhz: np.ndarray | float) -> np.ndarray:
	"""Convenience wrapper for ND02 frequency-to-temperature mapping."""
	return ND02.temperatures_k(frequencies_mhz)
