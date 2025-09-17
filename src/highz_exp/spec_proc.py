import os
from os.path import join as pjoin, basename as pbase
import copy
import numpy as np
import skrf as rf

def apply_to_all_ntwks(func, ntwk_dict):
    """Apply a function to the S-parameters of all networks in a dictionary."""
    new_ntwk_dict = {}
    for key, value in ntwk_dict.items():
        ntwk_copy = copy.deepcopy(ntwk_dict[key])
        spectrum = ntwk_copy.s[:, 0, 0]
        updated_spectrum = func(spectrum)
        ntwk_copy.s[:, 0, 0] = updated_spectrum
        new_ntwk_dict[key] = ntwk_copy
    return new_ntwk_dict

def remove_freq_range(ntwk, freq_band_to_remove) -> rf.Network:
    """Removes a specified frequency range from a skrf.Network object.

    Parameters:
    - ntwk (skrf.Network): The input network.
    - freq_band_to_remove (tuple): A tuple (min_freq, max_freq) specifying the frequency range to remove.
    """
    indices_to_keep = np.where((ntwk.f < freq_band_to_remove[0]) | (ntwk.f > freq_band_to_remove[1]))[0]
    new_ntwk = copy.deepcopy(ntwk)[indices_to_keep]
    return new_ntwk

def HP_filter(ntwk, faxis_hz, cutoff_hz):
    """High-pass filter: remove all frequencies below cutoff_hz."""
    filtered_data = remove_freq_range(ntwk, (0, cutoff_hz))
    faxis_filtered = faxis_hz[faxis_hz > cutoff_hz]
    return filtered_data, faxis_filtered

def interpolate_ntwk_dict(ntwk_dict, target_freqs, freq_range=None) -> dict:
    """
    Interpolate all ntwk objects in a dictionary to the target frequencies.

    Parameters:
    - ntwk_dict (dict): Dictionary of {'label': skrf.Network}
    - target_freqs (array-like): Frequencies to interpolate to (in Hz)
    - freq_range (tuple, optional): (min_freq, max_freq) to override common range

    Returns:
    - dict: New dictionary with deepcopied and interpolated skrf.Network objects
    """

    # Find the common frequency range across all networks
    common_min_freq = max(np.min(ntwk.f) for ntwk in ntwk_dict.values())
    common_max_freq = min(np.max(ntwk.f) for ntwk in ntwk_dict.values())
    
    # Use user-specified range if provided and within common range
    if freq_range is not None:
        min_freq, max_freq = freq_range
        if common_min_freq <= min_freq <= max_freq <= common_max_freq:
            clipped_freqs = np.clip(target_freqs, min_freq, max_freq)
        else:
            clipped_freqs = np.clip(target_freqs, common_min_freq, common_max_freq)
    else:
        clipped_freqs = np.clip(target_freqs, common_min_freq, common_max_freq)

    new_ntwk_dict = {}
    for label, ntwk in ntwk_dict.items():
        ntwk_copy = copy.deepcopy(ntwk)
        interp_ntwk = ntwk_copy.interpolate(clipped_freqs)
        new_ntwk_dict[label] = interp_ntwk
    return new_ntwk_dict