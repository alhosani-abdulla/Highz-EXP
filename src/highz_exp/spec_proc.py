import os
from os.path import join as pjoin, basename as pbase
import copy
import numpy as np
import skrf as rf

def subtract_s11_networks(ntwk1, ntwk2, new_name=None):
    """
    Create a new network where S11 = ntwk1.S11 - ntwk2.S11

    Parameters:
    - ntwk1 (skrf.Network): First network
    - ntwk2 (skrf.Network): Second network to subtract
    - new_name (str, optional): Name for the new network. If None, auto-generates name.

    Returns:
    - skrf.Network: New network with S11 = ntwk1.S11 - ntwk2.S11
    """

    # Verify frequencies match
    if not np.allclose(ntwk1.f, ntwk2.f):
        raise ValueError("Network frequencies don't match!")

    # Verify both networks have the same dimensions
    if ntwk1.s.shape != ntwk2.s.shape:
        raise ValueError("Network S-parameter dimensions don't match!")

    # Create new network by copying the first one
    new_ntwk = copy.deepcopy(ntwk1)

    # Subtract S11 parameters
    new_ntwk.s[:, 0, 0] = ntwk1.s[:, 0, 0] - ntwk2.s[:, 0, 0]

    # Set the name
    if new_name is None:
        name1 = getattr(ntwk1, 'name', 'ntwk1')
        name2 = getattr(ntwk2, 'name', 'ntwk2')
        new_ntwk.name = f"{name1}_minus_{name2}"
    else:
        new_ntwk.name = new_name

    return new_ntwk

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
    Interpolate all ntwk objects in a dictionary to the target frequencies and remove frequencies outside the specified range.

    Parameters:
    - ntwk_dict (dict): Dictionary of {'label': skrf.Network}
    - target_freqs (array-like): Frequencies to interpolate to (in Hz)
    - freq_range (tuple, optional): (min_freq, max_freq) to override common range

    Returns:
    - dict: New dictionary with deepcopied and interpolated skrf.Network objects
            with frequencies outside freq_range removed
    """
    # Find the common frequency range across all networks
    common_min_freq = max(np.min(ntwk.f) for ntwk in ntwk_dict.values())
    common_max_freq = min(np.max(ntwk.f) for ntwk in ntwk_dict.values())

    # Determine the frequency range to use
    if freq_range is not None:
        min_freq, max_freq = freq_range
        if not (common_min_freq <= min_freq <= max_freq <= common_max_freq):
            print(f"Warning: Requested range {freq_range[0]/1e6:.1f}-{freq_range[1]/1e6:.1f} MHz is outside common range "
                  f"{common_min_freq/1e6:.1f}-{common_max_freq/1e6:.1f} MHz")
            # Clip the requested range to the available data range
            min_freq = max(min_freq, common_min_freq)
            max_freq = min(max_freq, common_max_freq)
    else:
        min_freq, max_freq = common_min_freq, common_max_freq

    new_ntwk_dict = {}
    for label, ntwk in ntwk_dict.items():
        # Get the actual frequency range for this specific network
        ntwk_min_freq = np.min(ntwk.f)
        ntwk_max_freq = np.max(ntwk.f)

        # Clip target frequencies to both the desired range AND the network's actual range
        effective_min = max(min_freq, ntwk_min_freq)
        effective_max = min(max_freq, ntwk_max_freq)

        # Filter target frequencies to the effective range
        mask = (target_freqs >= effective_min) & (target_freqs <= effective_max)
        clipped_freqs = target_freqs[mask]

        if len(clipped_freqs) == 0:
            print(f"Warning: No target frequencies within valid range for {label}")
            continue

        # Create a copy and interpolate to the clipped frequencies
        ntwk_copy = copy.deepcopy(ntwk)
        interp_ntwk = ntwk_copy.interpolate(clipped_freqs)
        new_ntwk_dict[label] = interp_ntwk

    return new_ntwk_dict
