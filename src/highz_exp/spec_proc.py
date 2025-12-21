from datetime import datetime
from typing import List
import numpy as np
import logging, statistics, copy
import skrf as rf
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from typing import Sequence, Union

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

def interpolate_arrs(target_freqs, arr_freq, arr) -> tuple[np.ndarray, float, float]:
    """
    Interpolate an array to target frequencies within the common frequency range.
    Parameters:
    - target_freqs (array-like): Frequencies to interpolate to (in Hz)
    - arr_freq (array-like): Original frequencies of the array (in Hz)
    - arr (array-like): Original array values to interpolate
    Returns:
    - interpolated_arr (np.ndarray): Interpolated array values at target frequencies
    - common_min (float): Minimum frequency of the common range
    - common_max (float): Maximum frequency of the common range"""
    min_freq = np.min(target_freqs)
    common_min = max(min_freq, np.min(arr_freq))
    max_freq = np.max(target_freqs)
    common_max = min(max_freq, np.max(arr_freq))
    
    interpolate_freq = arr_freq[(arr_freq >= common_min) & (arr_freq <= common_max)]
    interpolate_arr = arr[(arr_freq >= common_min) & (arr_freq <= common_max)]
    
    interpolated_arr = np.interp(target_freqs, interpolate_freq, interpolate_arr)
    return interpolated_arr, common_min, common_max

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

# Adapted from Marcus Bosca's code
def compile_heatmap_data(alltime_spectra: np.ndarray, timestamps: List[datetime]) -> None:
    """
    Organizes pre-loaded spectral data by date, sorts chronologically, 
    and prepares a heatmap visualization.

    Args:
        alltime_spectra: A 2D NumPy array of shape (N, M) containing power values.
        timestamps: A list of N datetime objects corresponding to each spectrum row.
        
    Adapted from Marcus's code.
    """
    
    if len(alltime_spectra) != len(timestamps):
        raise ValueError("The number of spectra must match the number of timestamps.")

    # --- 1. Identify Day Boundaries ---
    # Since we have datetime objects, we can detect a new day by 
    # checking when the date changes compared to the previous entry.
    day_split_indices = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i].date() != timestamps[i-1].date():
            day_split_indices.append(i)
    day_split_indices.append(len(timestamps))

    # --- 2. Per-Day Segmentation and Sorting ---
    daily_values = []
    daily_timestamps = []

    for d in range(len(day_split_indices) - 1):
        start, end = day_split_indices[d], day_split_indices[d+1]
        
        # Sort indices within the day segment chronologically
        segment_indices = list(range(start, end))
        sorted_indices = sorted(segment_indices, key=lambda i: timestamps[i])
        
        daily_values.append(alltime_spectra[sorted_indices, :])
        daily_timestamps.append([timestamps[i] for i in sorted_indices])

    if not daily_values:
        print("No valid daily segments found.")
        return

    # Set the active dataset to the first day found (index 0)
    active_data = daily_values[0]
    active_times = daily_timestamps[0]

    return active_data, active_times

def validate_spectra_dimensions(datetimes, faxis_mhz, spectra) -> bool:
    """
    Validates that the dimensions of the spectra matrix match the 
    time and frequency axes.
    """
    num_time_steps = len(datetimes)
    num_freq_bins = len(faxis_mhz)
    h, w = spectra.shape

    # Check for exact match
    if h == num_time_steps and w == num_freq_bins:
        return True
    
    # Check if the matrix is transposed (Freq x Time instead of Time x Freq)
    if h == num_freq_bins and w == num_time_steps:
        raise ValueError(
            f"Dimension Mismatch: Spectra is {h}x{w} (Freq x Time), "
            f"but expected {num_time_steps}x{num_freq_bins} (Time x Freq). "
            "Try transposing your spectra with spectra.T"
        )
    
    # General mismatch
    raise ValueError(
        f"Dimension Mismatch:\n"
        f"  - Datetimes length: {num_time_steps}\n"
        f"  - Faxis length: {num_freq_bins}\n"
        f"  - Spectra shape: {h}x{w}\n"
        "The spectra matrix must have rows = time and columns = frequency."
    )

def downsample_waterfall(datetimes, faxis, spectra, max_pts=2000, step_t=None, step_f=None):
    """
    Reduces waterfall size using peak-preservation (max-pooling).

    Args:
        datetimes (np.array): Original datetime array.
        faxis (np.array): Original frequency axis array.
        spectra (np.array): 2D power spectra matrix.
        max_pts (int): Target max resolution for both axes if steps are not provided.
        step_t (int, optional): Manual downsample factor for Time. Defaults to None.
        step_f (int, optional): Manual downsample factor for Frequency. Defaults to None.

    Returns:
        tuple: (downsampled_datetimes, downsampled_faxis, downsampled_spectra)
    """
    h_orig, w_orig = spectra.shape
    
    # Determine steps: Use manual override if provided, else auto-calculate
    st = step_t if step_t is not None else max(1, h_orig // max_pts)
    sf = step_f if step_f is not None else max(1, w_orig // max_pts)
    
    if st == 1 and sf == 1:
        logging.info("Downsampling skipped: Data already within resolution limits.")
        return datetimes, faxis, spectra

    logging.info(f"Downsampling: st={st}, sf={sf}. Original: {h_orig}x{w_orig}")

    # Process Time Axis (Rows)
    if st > 1:
        new_h = h_orig // st
        # Trim to even multiple for reshape
        spectra = spectra[:new_h * st, :]
        spectra = spectra.reshape(new_h, st, w_orig).max(axis=1)
        datetimes = datetimes[::st][:new_h]

    # Process Frequency Axis (Cols)
    h_curr, _ = spectra.shape
    if sf > 1:
        new_w = w_orig // sf
        spectra = spectra[:, :new_w * sf]
        spectra = spectra.reshape(h_curr, new_w, sf).max(axis=2)
        faxis = faxis[::sf][:new_w]

    h_f, w_f = spectra.shape
    reduction = (1 - (spectra.size / (h_orig * w_orig))) * 100
    logging.info(f"Final shape: {h_f}x{w_f} ({reduction:.1f}% reduction).")
        
    return datetimes, faxis, spectra

# Define types for better IDE support
DatetimeArray = Union[np.ndarray, Sequence[np.datetime64]]

def get_dynamic_bin_size(datetimes: DatetimeArray) -> int:
    """
    Calculates the most frequent time interval (mode) between timestamps.
    
    This function uses NumPy vectorization for performance and ensures a 
    minimum bin size of 2 seconds. If multiple modes exist, the smallest 
    interval is returned.

    Args:
        datetimes: A NumPy array or sequence of datetime objects. 
            Expected to be sorted chronologically.

    Returns:
        int: The most frequent interval in seconds (minimum 2).
    """
    if len(datetimes) < 2:
        return 2  # Default fallback

    # Vectorized calculation of differences in seconds
    # np.diff handles the subtraction across the whole array at once
    intervals = np.diff(datetimes).astype('timedelta64[s]').astype(int)
    
    try:
        # multimode handles both single mode and ties gracefully
        modes = statistics.multimode(intervals)
        bin_size = min(modes)
    except Exception as e:
        logging.error(f"Failed to calculate mode: {e}")
        return 2

    final_bin = int(max(2, bin_size))
    logging.info(f"The most common spacing between two spectra is {final_bin} seconds.")
        
    return final_bin
