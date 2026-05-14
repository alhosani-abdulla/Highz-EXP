import numpy as np
from scipy.constants import Boltzmann as k_B
import copy, logging
from datetime import datetime, timezone, tzinfo
import skrf as rf
from astropy.time import Time
from zoneinfo import ZoneInfo

# Define some helper functions
def sparam_to_dB(s11):
    """Convert S11 reflection coefficient to dB scale."""
    return 20 * np.log10(np.abs(s11))

def rfsoc_spec_to_dbm(spectrum, offset=-135):
    """Convert recorded spectrum from digital spectrometer to dBm with an offset obtained from calibration."""
    spectrum = np.array(spectrum)
    finalSpectrum = 10 * np.log10(spectrum)+offset
    return finalSpectrum

def dbm_to_milliwatt(spectrum):
    """Convert spectrum in dBm to power in Watts."""
    return 10**(spectrum/10)

def watt_to_dbm(spectrum):
    """Convert power in Watts to dBm."""
    return 10 * np.log10(spectrum * 1000)

def dbm_to_kelvin(spectrum, channel_width=25*1000):
    """Convert spectrum in dBm/channel_width to noise temperature. """
    spectrum = np.array(spectrum)
    return dbm_to_milliwatt(spectrum) * 10**(-3) / channel_width / k_B

def kelvin_to_dbm(spectrum, channel_width=25*1000):
    """Convert noise temperature to dBm/channel_width."""
    spectrum = np.array(spectrum)
    return 10 * np.log10(spectrum * k_B * channel_width * 10**3)

def norm_factor(psd_ref, temperature=300):
    """Calculate the normalization factor for psd_ref in dBm to temperature"""
    gain = temperature / dbm_to_kelvin(psd_ref)
    return gain

def convert_utc_list_to_local(
    utc_timestamps: list[datetime],
    local_timezone: tzinfo | str | None = None,
) -> np.ndarray:
    """
    Converts a list of naive UTC datetime objects to timezone-aware local datetimes.
    
    Args:
        utc_timestamps: List or array of UTC datetime objects
        local_timezone: Target timezone or timezone name (defaults to system local timezone)
    
    Returns:
        np.ndarray: Array of datetime objects in the specified timezone
    """
    if not isinstance(utc_timestamps, (list, np.ndarray)):
        raise TypeError("utc_timestamps must be a list or numpy array")
    
    if len(utc_timestamps) == 0:
        raise ValueError("utc_timestamps cannot be empty")
    
    if not all(isinstance(ts, datetime) for ts in utc_timestamps):
        raise TypeError("All elements in utc_timestamps must be datetime objects")
    
    if isinstance(local_timezone, str):
        try:
            local_timezone = ZoneInfo(local_timezone)
        except Exception as exc:
            raise ValueError(f"Unknown timezone name: {local_timezone}") from exc

    if local_timezone is None:
        local_timezone = datetime.now().astimezone().tzinfo
        logging.info(f"Using system local timezone: {local_timezone}")

    local_ts = np.array([
        ts.replace(tzinfo=timezone.utc).astimezone(local_timezone)
        for ts in utc_timestamps
    ])
    return local_ts

def sidereal_hours_from_utcs(utc_list: list[datetime], longitude):

    """Convert a list of UTC datetime objects to sidereal hours at the given longitude.
     Args:
         utc_list (list): List of UTC datetime objects.
         longitude (Longitude): Longitude for sidereal time calculation.
     Returns:
         np.array: Array of sidereal hours.
     """
    return np.array([
        Time(ts, scale="utc").sidereal_time("apparent", longitude=longitude).hour
        for ts in utc_list
    ])

def ENR_to_kelvin(enr_db, T_ref=290, T_off=300):
    """Convert ENR in dB to noise temperature in Kelvin."""
    enr_linear = 10 ** (enr_db / 10)
    T_on = (1 + enr_linear) * T_ref
    return T_on

def kelvin_to_ENR(kelvin, T_ref=290):
    """Convert noise temperature in Kelvin to Noise Figure in dB."""
    return 10 * np.log10(kelvin / T_ref - 1)