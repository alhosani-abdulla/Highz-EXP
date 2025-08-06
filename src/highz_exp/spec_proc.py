import os
from os.path import join as pjoin, basename as pbase
import copy
import numpy as np
import skrf as rf

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