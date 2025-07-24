import numpy as np
from scipy.signal import find_peaks
from scipy.constants import Boltzmann as k_B

# Define some helper functions
def dbm_convert(spectrum, offset=-135):
    """Convert recorded spectrum to dBm with an offset obtained from calibration."""
    spectrum = np.array(spectrum)
    finalSpectrum = 10 * np.log10(spectrum)+offset
    return finalSpectrum

def dbm_to_power(spectrum):
    """Convert spectrum in dBm to power in Watts."""
    spectrum = np.array(spectrum)
    return 10**(spectrum/10)

def kelvin_convert(spectrum, channel_width=25*1000):
    """Convert spectrum in dBm/channel_width to noise temperature. """
    spectrum = np.array(spectrum)
    return dbm_to_power(spectrum) * 10**(-3) / channel_width / k_B

def norm_factor(psd_ref, temperature=300):
    """Calculate the normalization factor for psd_ref in dBm to temperature"""
    gain = temperature / kelvin_convert(psd_ref)
    return gain
