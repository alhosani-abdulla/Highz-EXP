from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import datetime as dt
import pickle

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator, interp1d
import skrf as rf
import healpy as hp
import pygdsm
import logging

from highz_exp.file_load import DSFileLoader

@dataclass
class SystemCalibrationProcessor:
	"""Shared calibration processing pipeline.

	This base class provides state handling and calibration helpers that are
	common across instruments. Instrument-specific subclasses must implement
	state loading and frequency-axis construction.
	"""

	min_frequency_mhz: float = 25.0
	max_frequency_mhz: float = 220.0
	site_latitude_deg: float = 51.8
	site_longitude_deg: float = -176.6
	site_elevation_m: float = 5.0
	state_list: list[str] = field(
		default_factory=lambda: [
			"antenna",
			"open_circuit",
			"short_circuit",
			"long_cable",
			"blackbody",
			"resistor",
			"noise_diode",
		]
	)

	raw_states: dict[str, dict[str, Any]] = field(default_factory=dict)
	state_power: dict[str, np.ndarray] = field(default_factory=dict)
	sidereal_time: dict[str, Any] = field(default_factory=dict)
	state_medians: dict[str, np.ndarray] = field(default_factory=dict)

	total_frequencies_mhz: np.ndarray | None = None
	frequencies_mhz: np.ndarray | None = None
	frequency_idx_range: np.ndarray | None = None

	system_gain: np.ndarray | None = None
	system_temp: np.ndarray | None = None
	system_gain_med: np.ndarray | None = None
	system_temp_med: np.ndarray | None = None
	calibration_cycle_ids: np.ndarray | None = None
	nd_temp: np.ndarray | None = None

	gain_freqs_mhz: np.ndarray | None = None
	gain_thetas_rad: np.ndarray | None = None
	gain_phis_rad: np.ndarray | None = None
	gain_pattern: np.ndarray | None = None

	def __post_init__(self) -> None:
		self.site_location = EarthLocation(
			lat=self.site_latitude_deg * u.deg,
			lon=self.site_longitude_deg * u.deg,
			height=self.site_elevation_m * u.m,
		)

	def save_pickle(
		self,
		file_path: str | Path,
		protocol: int = pickle.HIGHEST_PROTOCOL,
	) -> Path:
		"""Serialize this processor instance to a pickle file.

		Notes
		-----
		Pickle files are not secure against untrusted input. Only load files
		you created yourself or trust from a reliable source.
		"""
		path = Path(file_path).expanduser()
		path.parent.mkdir(parents=True, exist_ok=True)

		with path.open("wb") as fh:
			pickle.dump(self, fh, protocol=protocol)

		return path

	@classmethod
	def load_pickle(cls, file_path: str | Path):
		"""Load a previously pickled processor instance from disk."""
		path = Path(file_path).expanduser()
		if not path.exists():
			raise FileNotFoundError(f"Pickle file does not exist: {path}")

		with path.open("rb") as fh:
			obj = pickle.load(fh)

		if not isinstance(obj, cls):
			raise TypeError(
				f"Pickle at {path} contains {type(obj).__name__}, expected {cls.__name__}."
			)

		# Rebuild derived site location for older pickles if needed.
		if not hasattr(obj, "site_location"):
			obj.__post_init__()

		if not hasattr(obj, "raw_states"):
			logging.warning("Object from %s has not loaded any data.", path)

		return obj

	def prepare_state_medians(self) -> np.ndarray:
		"""Prepare frequency axis, slice states, and compute per-state medians."""
		frequencies_mhz = self.prepare_frequency_axis()
		self.slice_state_frequency_range()
		self.compute_state_medians()
		return frequencies_mhz

	def load_states(self, *args, **kwargs) -> dict[str, dict[str, Any]]:
		"""Placeholder for state loading logic. To be implemented in subclass."""
		raise NotImplementedError("load_states() must be implemented in a subclass.")

	def prepare_frequency_axis(self) -> np.ndarray:
		"""Build instrument-specific frequency axis and selected range."""
		raise NotImplementedError("prepare_frequency_axis() must be implemented in a subclass.")

	def load_resistor_median_for_segment(
		self,
		data_folder: str | Path,
		date: str,
		no_segments: int,
		seg_indx: int,
		resistor_state_no: int = 5,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Load one segment of resistor-state spectra and return timestamps + median."""
		if self.frequency_idx_range is None:
			self.prepare_frequency_axis()

		time_dirs = DSFileLoader.get_sorted_time_dirs(str(data_folder))
		resistor_time_dirs = np.array_split(time_dirs, no_segments)[seg_indx]
		resistor_loaded = DSFileLoader.load_and_add_timestamps(
			date,
			list(resistor_time_dirs),
			resistor_state_no,
		)
		timestamps, spectra, _ = DSFileLoader.read_loaded(
			resistor_loaded,
			sort="ascending",
			convert=False,
		)
		resistor_median = np.median(spectra[:, self.frequency_idx_range], axis=0)
		return timestamps, resistor_median

	@staticmethod
	def build_segment_local_label(local_timestamps, seg_indx: int, timezone_name: str = "HST") -> str:
		"""Build a compact segment label from local timestamps for plotting legends."""
		if len(local_timestamps) == 0:
			return f"Seg {seg_indx}"
		start_local = local_timestamps[0]
		end_local = local_timestamps[-1]
		start_str = start_local.strftime("%m-%d %H:%M")
		if start_local.date() == end_local.date():
			end_str = end_local.strftime("%H:%M")
		else:
			end_str = end_local.strftime("%m-%d %H:%M")
		return f"Seg {seg_indx} {start_str}-{end_str} {timezone_name}"

	@staticmethod
	def choose_frequency_downsample_step(frequency_bin_count: int, requested_step: int) -> int:
		"""Return the largest valid downsample divisor <= requested_step."""
		valid_steps = [
			divisor for divisor in range(1, requested_step + 1)
			if frequency_bin_count % divisor == 0
		]
		return max(valid_steps) if valid_steps else 1

	def slice_state_frequency_range(self) -> dict[str, np.ndarray]:
		"""Slice each state spectra to the configured frequency range."""
		if self.frequency_idx_range is None:
			self.prepare_frequency_axis()

		for name in self.state_list:
			state = self.raw_states.get(name)
			if state is None:
				logging.warning("State '%s' not found in loaded states; skipping frequency slicing.", name)
				continue
			self.state_power[name] = state["spectra"][:, self.frequency_idx_range]
		return self.state_power

	def compute_sidereal_timestamps(self) -> dict[str, Any]:
		"""Compute apparent sidereal timestamps for each loaded state."""
		sidereal: dict[str, Any] = {}
		for name in self.state_list:
			state = self.raw_states.get(name)
			if state is None or len(state["timestamps"]) == 0:
				sidereal[name] = np.array([])
				continue
			t = Time(state["timestamps"], scale="utc")
			sidereal[name] = t.sidereal_time("apparent", longitude=self.site_location.lon)
		self.sidereal_time = sidereal
		return sidereal

	@staticmethod
	def frequency_ticks(frequencies_mhz: np.ndarray, step_mhz: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
		"""Get x-axis tick indices/labels for a frequency axis. 
		This doesn't rebin the data, just identifies where to place ticks for plotting."""
		idx = np.where(np.diff((frequencies_mhz // step_mhz) * step_mhz) != 0)[0] + 1
		return idx, frequencies_mhz[idx]

	@staticmethod
	def plot_ticks_from_hours(hours: np.ndarray, step_hour: int = 1) -> tuple[np.ndarray, np.ndarray]:
		"""Get tick indices/labels from hour values."""
		idx = np.where(np.diff((hours // step_hour) * step_hour) != 0)[0] + 1
		labels = np.array(hours[idx], dtype=int)
		return idx, labels

	def compute_time_ticks_by_state(self, step_hour: int = 1) -> dict[str, dict[str, np.ndarray]]:
		"""Compute sidereal time tick indices/labels for each state."""
		if not self.sidereal_time:
			self.compute_sidereal_timestamps()

		output: dict[str, dict[str, np.ndarray]] = {}
		for name, sidereal in self.sidereal_time.items():
			if len(sidereal) == 0:
				output[name] = {"idx": np.array([]), "labels": np.array([])}
				continue
			idx, labels = self.plot_ticks_from_hours(sidereal.hour, step_hour=step_hour)
			output[name] = {"idx": idx, "labels": labels}
		return output

	def compute_state_medians(self) -> dict[str, np.ndarray]:
		"""Compute median spectrum for each loaded state for all samples, erase cycle resolution."""
		medians: dict[str, np.ndarray] = {}
		for name, data in self.state_power.items():
			medians[name] = np.median(data, axis=0)
			logging.info("Computed median for state '%s' with shape %s", name, medians[name].shape)
		self.state_medians = medians
		return medians

	def _iter_state_cycle_medians(self):
		"""Yield per-state cycle medians as (state_name, cycle_no, cycle_median)."""
		for name, data in self.raw_states.items():
			cycles = data.get("cycles")
			spectra = self.state_power.get(name)
			if spectra is None:
				logging.warning("No sliced spectra found for state '%s'; skipping.", name)
				continue
			if cycles is None:
				logging.warning("No cycle array found for state '%s'; skipping.", name)
				continue

			for cycle in np.unique(cycles):
				cycle_mask = cycles == cycle
				if not np.any(cycle_mask):
					logging.warning("No samples found for state '%s' cycle %s; skipping.", name, cycle)
					continue
				yield name, cycle, np.median(spectra[cycle_mask], axis=0)

	def _noise_diode_temperature_profile(
		self,
		temp_func,
	) -> np.ndarray:
		"""Set noise diode temperature profile"""
		if self.frequencies_mhz is None:
			self.prepare_frequency_axis()
		
		self.nd_temp = temp_func(self.frequencies_mhz)
		return self.nd_temp

	def compute_state_medians_by_cycle(self) -> dict[str, list[np.ndarray]]:
		"""Compute median spectrum for each state separately for each cycle."""
		cycle_medians: dict[str, list[np.ndarray]] = {}
		for name, _, cycle_median in self._iter_state_cycle_medians():
			cycle_medians.setdefault(name, []).append(cycle_median)

		for name, medians in cycle_medians.items():
			logging.info("Computed %d cycle medians for state '%s'.", len(medians), name)
		return cycle_medians

	def calibrate_system_from_medians(
		self,
		noise_diode_temp_f1_k: float = 2086.0,
		noise_diode_temp_f2_k: float = 2002.0,
		noise_diode_freq_f1_mhz: float = 50.0,
		noise_diode_freq_f2_mhz: float = 200.0,
		resistor_temp_k: float = 275.0,
		resistor_median: np.ndarray | None = None,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Compute system gain and system temperature from median diode/resistor spectra.

		Noise diode temperature is modeled as a linear function of frequency defined
		by two anchor points:
		(noise_diode_freq_f1_mhz, noise_diode_temp_f1_k) and
		(noise_diode_freq_f2_mhz, noise_diode_temp_f2_k).
		"""
		if self.frequencies_mhz is None:
			self.prepare_frequency_axis()
		if not self.state_medians:
			self.compute_state_medians()
		if noise_diode_freq_f1_mhz == noise_diode_freq_f2_mhz:
			raise ValueError("noise_diode_freq_f1_mhz and noise_diode_freq_f2_mhz must be different.")

		self.nd_temp = noise_diode_temp_f1_k + (
			(noise_diode_temp_f2_k - noise_diode_temp_f1_k)
			/ (noise_diode_freq_f2_mhz - noise_diode_freq_f1_mhz)
		) * (self.frequencies_mhz - noise_diode_freq_f1_mhz)

		if "noise_diode" not in self.state_medians:
			raise ValueError("'noise_diode' median is required for calibration.")
		noise_diode = self.state_medians["noise_diode"]
		if resistor_median is None:
			if "resistor" not in self.state_medians:
				raise ValueError("'resistor' median missing. Provide resistor_median or load resistor state.")
			resistor = self.state_medians["resistor"]
		else:
			resistor = np.asarray(resistor_median)

		self.system_gain_med = (noise_diode - resistor) / (self.nd_temp - resistor_temp_k)
		self.system_temp_med = resistor / self.system_gain_med - resistor_temp_k
		return self.system_gain_med, self.system_temp_med

	def calibrate_system_from_cycles(self, noise_diode_temp_func,
		resistor_temp_k: float = 275.0,
	) -> dict[str, list[tuple[np.ndarray, np.ndarray | float | None]]]:
		"""Compute per-cycle calibration tuples and aggregate gain/temperature.

		The returned dictionary preserves per-cycle median products for each state.
		Additionally, this method computes per-cycle system gain/temperature from
		noise-diode and resistor cycles and stores them as 2D arrays with shape
		``(n_cycle, n_frequency)`` into ``self.system_gain`` and
		``self.system_temp``.
		 - noise_diode_temp_func: A function that takes frequencies_mhz and returns the noise diode temperature profile in Kelvin.
		 - resistor_temp_k: The physical temperature of the resistor load in Kelvin.
		"""
		if self.frequencies_mhz is None:
			self.prepare_frequency_axis()
		if not self.state_power:
			self.slice_state_frequency_range()
		nd_temp_profile = self._noise_diode_temperature_profile(noise_diode_temp_func)

		cycle_calibrations: dict[str, list[tuple[np.ndarray, np.ndarray | float | None]]] = {}
		for name, _, cycle_median in self._iter_state_cycle_medians():
			if name == "noise_diode":
				cal_value: np.ndarray | float | None = nd_temp_profile
			elif name == "resistor":
				cal_value = resistor_temp_k
			else:
				cal_value = None
			cycle_calibrations.setdefault(name, []).append((cycle_median, cal_value))

		for name, values in cycle_calibrations.items():
			logging.info("Computed cycle calibrations for state '%s' with %d cycles.", name, len(values))

		noise_diode_cycles = cycle_calibrations.get("noise_diode", [])
		resistor_cycles = cycle_calibrations.get("resistor", [])
		if not noise_diode_cycles or not resistor_cycles:
			raise ValueError("Cycle calibration requires both 'noise_diode' and 'resistor' states.")

		noise_diode_cycle_ids = np.unique(self.raw_states["noise_diode"]["cycles"])
		resistor_cycle_ids = np.unique(self.raw_states["resistor"]["cycles"])
		paired_cycle_ids = np.intersect1d(noise_diode_cycle_ids, resistor_cycle_ids)
		pair_count = len(paired_cycle_ids)
		if pair_count == 0:
			raise ValueError("No overlapping cycle IDs found between 'noise_diode' and 'resistor'.")

		if len(noise_diode_cycle_ids) != len(resistor_cycle_ids):
			logging.warning(
				"Cycle count mismatch: noise_diode=%d, resistor=%d. Using %d overlapping cycle(s).",
				len(noise_diode_cycle_ids),
				len(resistor_cycle_ids),
				pair_count,
			)

		cycle_gains = []
		cycle_temps = []
		noise_diode_power = self.state_power["noise_diode"]
		resistor_power = self.state_power["resistor"]
		noise_diode_cycle_labels = self.raw_states["noise_diode"]["cycles"]
		resistor_cycle_labels = self.raw_states["resistor"]["cycles"]

		for cycle_id in paired_cycle_ids:
			noise_diode_median = np.median(noise_diode_power[noise_diode_cycle_labels == cycle_id], axis=0)
			resistor_median = np.median(resistor_power[resistor_cycle_labels == cycle_id], axis=0)
			gain = (noise_diode_median - resistor_median) / (nd_temp_profile - resistor_temp_k)
			temp = resistor_median / gain - resistor_temp_k
			cycle_gains.append(gain)
			cycle_temps.append(temp)

		# Keep cycle-resolved calibration instead of collapsing over cycles.
		self.system_gain = np.asarray(cycle_gains)
		self.system_temp = np.asarray(cycle_temps)
		self.calibration_cycle_ids = np.asarray(paired_cycle_ids)
		self.nd_temp = nd_temp_profile

		return cycle_calibrations

	def calibrated_temperature(self, state_name: str) -> np.ndarray:
		"""Return median calibrated temperature from per-sample calibrated 2D state power."""
		calibrated_2d = self.calibrate_2d_state_power(state_name)
		return np.median(calibrated_2d, axis=0)

	def calibrate_2d_state_power(self, state_name: str) -> np.ndarray:
		"""Convert a state's 2D power array into calibrated temperature."""
		if self.system_gain is None or self.system_temp is None:
			self.calibrate_system_from_medians()

		state_power = self.state_power[state_name]
		gain = self.system_gain
		temp = self.system_temp

		if gain.ndim == 1:
			return state_power / gain - temp

		state = self.raw_states.get(state_name)
		if state is None or "cycles" not in state:
			raise ValueError(f"State '{state_name}' missing cycle labels required for cycle-resolved calibration.")

		if self.calibration_cycle_ids is None:
			raise ValueError("calibration_cycle_ids not set. Run calibrate_system_from_cycles() first.")

		cycles = np.asarray(state["cycles"])
		if len(cycles) != state_power.shape[0]:
			raise ValueError(
				f"Cycle label length mismatch for state '{state_name}': "
				f"{len(cycles)} labels for {state_power.shape[0]} spectra."
			)

		cycle_to_row = {cycle_id: i for i, cycle_id in enumerate(self.calibration_cycle_ids)}
		row_idx = np.array([cycle_to_row.get(cycle, -1) for cycle in cycles], dtype=int)
		if np.any(row_idx < 0):
			missing = np.unique(cycles[row_idx < 0])
			raise ValueError(
				f"State '{state_name}' contains cycle IDs not present in calibration: {missing.tolist()}"
			)

		return state_power / gain[row_idx, :] - temp[row_idx, :]

	@staticmethod
	def load_s11_mismatch(
		antenna_response_file: str | Path,
		target_frequencies_mhz: np.ndarray,
		interp_kind: str = "linear",
	) -> tuple[np.ndarray, np.ndarray]:
		"""Load S11 file and interpolate mismatch loss onto target frequencies."""
		ntwk = rf.Network(str(antenna_response_file))
		s11 = ntwk.s[:, 0, 0]
		mismatch_loss = 1 - np.abs(s11) ** 2
		nwk_frequencies_mhz = ntwk.f / 1e6
		f_interp = interp1d(nwk_frequencies_mhz, mismatch_loss, kind=interp_kind, fill_value="extrapolate")
		mismatch_interp = f_interp(target_frequencies_mhz)
		return mismatch_interp, nwk_frequencies_mhz

	@staticmethod
	def apply_mismatch_correction(
		antenna_calibrated: np.ndarray,
		mismatch_interp: np.ndarray,
	) -> np.ndarray:
		"""Apply mismatch correction to a 1D or 2D calibrated temperature array."""
		return antenna_calibrated / mismatch_interp

	@staticmethod
	def interpolate_waterfall_grid(
		utc_timestamps: list[dt.datetime] | np.ndarray,
		calibrated_waterfall: np.ndarray,
		frequencies_mhz: np.ndarray,
		new_frequencies_mhz: np.ndarray,
		step_seconds: int = 600,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Interpolate calibrated waterfall onto a regular time/frequency grid."""
		t0 = utc_timestamps[0]
		t_sec = np.array([(ti - t0).total_seconds() for ti in utc_timestamps])

		positive = np.where(np.diff(t_sec) > 0)[0]
		t_sec = t_sec[positive]
		waterfall = calibrated_waterfall[positive]

		interpolator = RegularGridInterpolator((t_sec, frequencies_mhz), waterfall)
		t_new = np.arange(t_sec[0], t_sec[-1], step_seconds)
		new_utc = np.array([t0 + dt.timedelta(seconds=float(s)) for s in t_new])

		out = np.zeros((len(t_new), len(new_frequencies_mhz)))
		for i, t in enumerate(t_new):
			points = np.column_stack((np.full_like(new_frequencies_mhz, t, dtype=float), new_frequencies_mhz))
			out[i, :] = interpolator(points)

		return out, new_utc, t_new

	def load_gain_pattern_from_csv(
		self,
		csv_path: str | Path,
		gain_freqs_mhz: np.ndarray | None = None,
		skiprows: int = 2,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""Load antenna gain/effective-height table from CSV."""
		gain_info = np.loadtxt(str(csv_path), delimiter=",", skiprows=skiprows)
		if gain_freqs_mhz is None:
			gain_freqs_mhz = np.arange(25, 250 + 25, 25)

		self.gain_freqs_mhz = np.asarray(gain_freqs_mhz, dtype=float)
		self.gain_thetas_rad = np.deg2rad(gain_info[:, 0])
		self.gain_phis_rad = np.linspace(0, 2 * np.pi, 181, endpoint=True)
		self.gain_pattern = gain_info[:, 1:].T
		return self.gain_freqs_mhz, self.gain_thetas_rad, self.gain_phis_rad, self.gain_pattern

	def load_effective_height(self, frequency_mhz: int | float) -> np.ndarray:
		"""Get 2D effective-height map at one modeled frequency."""
		if self.gain_pattern is None or self.gain_freqs_mhz is None:
			raise ValueError("Gain pattern not loaded. Call load_gain_pattern_from_csv() first.")

		idx = np.where(self.gain_freqs_mhz == frequency_mhz)[0]
		if len(idx) == 0:
			raise ValueError(f"Frequency {frequency_mhz} MHz not available in gain grid.")
		return self.gain_pattern[idx].repeat(181, axis=0).T

	def generate_beam_map(self, frequency_mhz: int | float, n_side: int = 256) -> np.ndarray:
		"""Generate HEALPix beam map from loaded effective-height model."""
		effective_height_2d_map = self.load_effective_height(frequency_mhz)
		num_theta, num_phi = effective_height_2d_map.shape
		theta = np.linspace(0, np.pi / 2, num_theta)
		phi = np.linspace(0, 2 * np.pi, num_phi)

		interpolator = RegularGridInterpolator(
			(theta, phi),
			effective_height_2d_map,
			bounds_error=False,
			fill_value=0,
		)

		n_pix = hp.nside2npix(n_side)
		beam_map = np.full(n_pix, 0.0)
		theta_hp, phi_hp = hp.pix2ang(n_side, np.arange(n_pix))
		mask_upper = theta_hp <= (np.pi / 2)
		interp_points = np.vstack((theta_hp[mask_upper], phi_hp[mask_upper])).T
		beam_map[mask_upper] = interpolator(interp_points)

		rot = hp.Rotator([0, 90, 0])
		return rot.rotate_map_pixel(beam_map)

	def generate_healpix_map(self, frequency_mhz: float, utc_timestamp: dt.datetime, observer: str = "LFSM") -> np.ndarray:
		"""Generate sky temperature map from PyGDSM for a timestamp/frequency."""
		if observer == "08":
			ov = pygdsm.GSMObserver08()
		elif observer == "16":
			ov = pygdsm.GSMObserver16()
		elif observer == "Haslam":
			ov = pygdsm.HaslamObserver()
		else:
			ov = pygdsm.LFSMObserver()

		ov.lon = self.site_longitude_deg
		ov.lat = self.site_latitude_deg
		ov.elev = self.site_elevation_m
		ov.date = utc_timestamp

		hmap = ov.generate(frequency_mhz)
		return np.ma.filled(hmap, fill_value=0)

	def create_simulated_waterfall(
		self,
		utc_timestamps: list[dt.datetime] | np.ndarray,
		frequencies_mhz: np.ndarray,
		n_side: int = 256,
		observer: str = "LFSM",
	) -> np.ndarray:
		"""Compute simulated waterfall via beam-squared sky integral."""
		omega = hp.nside2pixarea(n_side)
		simulated = np.zeros((len(utc_timestamps), len(frequencies_mhz)))

		for j, freq in enumerate(frequencies_mhz):
			beam_map = self.generate_beam_map(int(freq), n_side=n_side)
			for i, timestamp in enumerate(utc_timestamps):
				hmap = self.generate_healpix_map(float(freq), timestamp, observer=observer)
				simulated[i, j] = np.sum(hmap * beam_map**2) * omega

		return simulated

@dataclass
class DSCalibrationProcessor(
	SystemCalibrationProcessor
):
	"""Extended processor for digital spectrometer calibration with DS-specific loading."""

	num_frequency_samples: int = 16384
	frequency_bin_size_mhz: float = 0.025

	def prepare_frequency_axis(self) -> np.ndarray:
		"""Create DS frequency axis and selected analysis range."""
		self.total_frequencies_mhz = np.arange(self.num_frequency_samples) * self.frequency_bin_size_mhz
		frequency_idx_range = np.where(
			(self.total_frequencies_mhz >= self.min_frequency_mhz)
			& (self.total_frequencies_mhz <= self.max_frequency_mhz)
		)[0]

		if len(frequency_idx_range) == 0:
			raise ValueError(
				"No frequency bins found in selected range "
				f"[{self.min_frequency_mhz}, {self.max_frequency_mhz}] MHz."
			)

		# Ensure an even number of points for downstream processing assumptions.
		if len(frequency_idx_range) % 2 != 0:
			if len(frequency_idx_range) == 1:
				raise ValueError(
					"Selected frequency range contains only one bin; cannot enforce an even bin count."
				)
			dropped_idx = frequency_idx_range[-1]
			logging.info(
				"Selected frequency bin count is odd (%d). Dropping highest bin %.6f MHz to make it even.",
				len(frequency_idx_range),
				self.total_frequencies_mhz[dropped_idx],
			)
			frequency_idx_range = frequency_idx_range[:-1]

		self.frequency_idx_range = frequency_idx_range
		self.frequencies_mhz = self.total_frequencies_mhz[self.frequency_idx_range]
		return self.frequencies_mhz

	def load_states(
		self,
		data_folder: str | Path,
		convert: bool = False,
		no_segments: int = 1,
		seg_indx: int = 0,
		states_to_load: list[str] | None = None,
	) -> dict[str, dict[str, Any]]:
		"""Load all configured switch states using ``DSFileLoader``.

		Returns a dictionary keyed by state name with:
		- ``timestamps``: np.ndarray of UTC datetimes
		- ``spectra``: np.ndarray of raw/converted spectra

		Parameters
		----------
		convert : bool, optional
			Whether to convert raw spectra to power units during loading. If False, loads raw spectra.
		no_segments : int, optional
			Number of segments to split the day folder into.
		seg_indx : int, optional
			Zero-based index of the segment to load.
		states_to_load : list[str] | None, optional
			Subset of state names to load. If None, loads all states in ``state_list``.
		"""
		if no_segments < 1:
			raise ValueError("no_segments must be >= 1.")
		if not (0 <= seg_indx < no_segments):
			raise ValueError(f"seg_indx must satisfy 0 <= seg_indx < no_segments ({no_segments}).")
		if states_to_load is not None:
			unknown = sorted(set(states_to_load) - set(self.state_list))
			if unknown:
				raise ValueError(f"Unknown states requested: {unknown}")
			state_filter = set(states_to_load)
		else:
			state_filter = set(self.state_list)

		data_folder = Path(data_folder)
		date = data_folder.name
		if not date or not date.isdigit() or len(date) != 8:
			raise ValueError(
				f"Unable to infer date from input directory '{data_folder}'. "
				"Expected a day folder named YYYYMMDD (e.g., /path/to/20260303)."
			)

		time_dirs = DSFileLoader.get_sorted_time_dirs(str(data_folder))
		if len(time_dirs) == 0:
			raise ValueError(f"No time folders found under: {data_folder}")

		segmented_time_dirs = np.array_split(time_dirs, no_segments)[seg_indx]
		if len(segmented_time_dirs) == 0:
			raise ValueError(
				f"Selected segment {seg_indx} is empty for no_segments={no_segments}."
			)

		loaded: dict[str, dict[str, Any]] = {}
		for state_no, name in enumerate(self.state_list):
			if name not in state_filter:
				continue
			state_loaded = DSFileLoader.load_and_add_timestamps(
				date,
				list(segmented_time_dirs),
				state_no,
			)
			timestamps, spectra, cycles = DSFileLoader.read_loaded(
				state_loaded,
				sort="ascending",
				convert=convert,
			)
			loaded[name] = {"timestamps": timestamps, "spectra": spectra, "cycles": cycles}
		self.raw_states = loaded
		return loaded
