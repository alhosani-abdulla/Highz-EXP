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
    return 10 * np.log10(spectrum) + 30

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

def remove_freq_range(ntwk, freq_range_to_remove):
    """Removes a specified frequency range from a skrf.Network object.

    Parameters:
    - ntwk (skrf.Network): The input network.
    - freq_range_to_remove (tuple): A tuple (min_freq, max_freq) specifying the frequency range to remove.

    Returns:
    - skrf.Network: A new network with the specified frequency range removed.
    """
    # Find the indices of the frequencies to keep
    indices_to_keep = np.where((ntwk.f < freq_range_to_remove[0]) | (ntwk.f > freq_range_to_remove[1]))[0]

    # Create a new network with the filtered data
    new_ntwk = copy.deepcopy(ntwk)[indices_to_keep]
    
    return new_ntwk

def interpolate_ntwk_dict(ntwk_dict, target_freqs):
    """
    Interpolate all ntwk objects in a dictionary to the target frequencies.

    Parameters:
    - ntwk_dict (dict): Dictionary of {'label': skrf.Network}
    - target_freqs (array-like): Frequencies to interpolate to (in Hz)

    Returns:
    - dict: New dictionary with deepcopied and interpolated skrf.Network objects
    """

    new_ntwk_dict = {}
    for label, ntwk in ntwk_dict.items():
        ntwk_copy = copy.deepcopy(ntwk)
        interp_ntwk = ntwk_copy.interpolate(target_freqs)
        new_ntwk_dict[label] = interp_ntwk
    return new_ntwk_dict
