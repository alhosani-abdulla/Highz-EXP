import copy
import numpy as np
import skrf as rf
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

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

def smooth_spectrum(data, method='savgol', window=31):
    """
    Return a smoothed copy of `data` using the requested method.
    Safe-guards window size to be appropriate for the data length.
    """
    n = len(data)
    w = int(window)

    if method == 'savgol':
        # enforce reasonable window size and oddness
        if w < 3:
            w = 3
        if w % 2 == 0:
            w += 1
        if w > n:
            # reduce to largest odd <= n
            w = n if n % 2 == 1 else n - 1
        if w < 3:
            return data.copy()
        return savgol_filter(data, w, polyorder=3)

    elif method == 'moving_avg':
        if w < 1:
            w = 1
        return uniform_filter1d(data, size=w, mode='nearest')

    else:
        # unknown method -> return original
        return data.copy()

def remove_spikes_from_psd(freq, psd, threshold=5.0, window=5):
    """
    Removes spike-like peaks in the PSD by detecting outliers and interpolating over them.

    Parameters:
        freq: np.ndarray
            Frequency values (same shape as psd).
        psd: np.ndarray
            PSD values (V²/Hz or similar).
        threshold: float
            How many times the local MAD a point must exceed to be considered a spike.
        window: int
            Number of points on each side to use in local median filtering.

    Returns:
        psd_cleaned: np.ndarray
            PSD with spikes removed (replaced via interpolation).
    """
    from scipy.ndimage import median_filter

    psd = np.asarray(psd)
    smoothed = median_filter(psd, size=2 * window + 1)

    # Residuals and thresholding
    residual = psd - smoothed
    mad = np.median(np.abs(residual))  # median absolute deviation
    spike_mask = np.abs(residual) > threshold * mad

    # Replace spikes by interpolation
    psd_cleaned = copy.deepcopy(psd)
    spike_indices = np.where(spike_mask)[0]
    keep_indices = np.where(~spike_mask)[0]

    if len(keep_indices) >= 2:
        psd_cleaned[spike_mask] = np.interp(freq[spike_mask], freq[keep_indices], psd[keep_indices])

    return psd_cleaned
   
def despike(arr, window_len: int = 11, threshold: float = 5.0, replace: str = "median") -> np.ndarray:
    """
    Remove narrow RFI spikes by comparing each point to a local median and MAD.

    Parameters:
        window_len: odd integer window size for local statistics (>=3).
        threshold: multiple of local MAD (median absolute deviation) above which a point is considered a spike.
        replace: 'median' to replace spikes with local median, 'interp' to interpolate
                    across spike points using neighboring good points.

    Notes:
        This uses numpy's sliding_window_view when available, or scipy.signal.medfilt
        as a fallback. Both scipy.signal.medfilt and numpy.lib.stride_tricks.sliding_window_view
        can be used to speed up the local-median computation.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.signal import medfilt

    if window_len < 3:
        return arr
    wl = int(window_len)
    if wl % 2 == 0:
        wl += 1
    pad = wl // 2

    # try fast sliding-window median/MAD
    try:

        padded = np.pad(arr, pad, mode="edge")
        windows = sliding_window_view(padded, wl)
        local_med = np.median(windows, axis=1)
        local_mad = np.median(np.abs(windows - local_med[:, None]), axis=1)
    except Exception:
        # fallback: try scipy medfilt for median; compute MAD with small local loops
        try:

            local_med = medfilt(arr, kernel_size=wl)
            local_mad = np.empty_like(arr)
            n = arr.size
            for i in range(n):
                i0 = max(0, i - pad)
                i1 = min(n, i + pad + 1)
                w = arr[i0:i1]
                m = np.median(w)
                local_mad[i] = np.median(np.abs(w - m))
        except Exception:
            # last resort: global median/MAD
            gm = np.median(arr)
            gmad = np.median(np.abs(arr - gm)) or 1.0
            local_med = np.full_like(arr, gm)
            local_mad = np.full_like(arr, gmad)

    # detect spikes using MAD (robust to outliers)
    local_mad_safe = np.where(local_mad <= 0, 1e-12, local_mad)
    diff = np.abs(arr - local_med)
    spikes = diff > (threshold * local_mad_safe)
    if not np.any(spikes):
        return arr

    if replace == "median":
        arr = np.where(spikes, local_med, arr)
    elif replace == "interp":
        x = np.arange(len(arr))
        good = (~spikes) & (~np.isnan(arr))
        if good.sum() < 2:
            # not enough points to interpolate, fallback to median replacement
            arr = np.where(spikes, local_med, arr)
        else:
            new = arr.copy()
            new[spikes] = np.interp(x[spikes], x[good], arr[good])
            arr = new
    else:
        raise ValueError("replace must be 'median' or 'interp'")

    return arr

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
