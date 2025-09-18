import numpy as np
from scipy.signal import find_peaks
from scipy.constants import Boltzmann as k_B
import copy
import skrf as rf

# Define some helper functions
def s11_to_dB(s11):
    """Convert S11 reflection coefficient to dB scale."""
    return 20 * np.log10(np.abs(s11))

def spec_to_dbm(spectrum, offset=-135):
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
