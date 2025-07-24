import numpy as np
import os, glob, copy
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from scipy.constants import Boltzmann as k_B
from .unit_convert import dbm_convert, dbm_to_power, kelvin_convert

pjoin = os.path.join

def get_and_clean_nonempty_files(directory, pattern="*.npy"):
    """
    Return a list of non-zero-size files in `directory` matching `pattern`,
    and remove all 0-byte files.

    Parameters:
        directory (str): Path to the directory (mounted Google Drive path).
        pattern (str): Glob pattern, e.g., '*.npy', '*.txt', etc.

    Returns:
        List[str]: List of full paths to non-zero-length files.
    """
    all_files = glob.glob(os.path.join(directory, pattern))
    nonempty_files = []

    for f in all_files:
        if os.path.getsize(f) > 0:
            nonempty_files.append(f)
        else:
            print(f"Removing empty file: {f}")
            os.remove(f)

    return nonempty_files

def count_spikes(y, height=None, threshold=None, distance=None):
    """Count the number of peaks in a signal that exceed a specified height and threshold."""
    peaks, _ = find_peaks(y, height=height, threshold=threshold, distance=distance)
    return len(peaks)

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

def load_npy(dir_path, pattern='*state*.npy'):
    """Load all non-empty .npy files from a specified directory and return them as a list"""
    data_full_path = pjoin(dir_path)
    state_files = get_and_clean_nonempty_files(data_full_path, pattern)
    states = []
    for file in state_files:
        states.append(np.load(file, allow_pickle=True).item())
    return states

def load_npy_spec(dir_path, pick_snapshot=None):
    """Load all non-empty .npy files from a specified relative directory path and return them as a list of states."""
    data_full_path = pjoin(dir_path)
    state_files = []
    _ = get_and_clean_nonempty_files(data_full_path, f'*state1*.npy')
    _ = get_and_clean_nonempty_files(data_full_path, f'*state0*.npy')
    for i in range(2, 8):
        state_files.append(get_and_clean_nonempty_files(data_full_path, f'*state{i}*.npy'))
    state_files.append(get_and_clean_nonempty_files(data_full_path, "*stateOC*.npy"))

    load_states = []
    for i, file_list in enumerate(state_files):
        if isinstance(pick_snapshot, list):
            load_states.append(np.load(file_list[pick_snapshot[i]], allow_pickle=True).item())
        elif pick_snapshot is not None:
            spikes = [count_spikes(dbm_convert(np.load(file, allow_pickle=True).item()['spectrum']), height=20) for file in file_list]
            if min(spikes) > 0:
                print(f"All the recorded spectrum files for state{i+2} have spikes with more than 20 dB of height")
            indx_least_spikes = np.argmin(spikes)
            load_states.append(np.load(file_list[indx_least_spikes], allow_pickle=True).item())
        else:
            load_states.append(np.load(file_list[0], allow_pickle=True).item())

    return load_states

def preprocess_states(faxis, df, load_states, remove_spikes=True, unit='dBm', offset=-135, system_gain=100, normalize=None):
    """Preprocess the loaded states by converting the spectrum to the specified unit and removing spikes if required
    
    Parameters:
        faxis: np.ndarray, frequency points in Hz. 
        df: in MHz, the channel width."""
    loaded_states_copy = copy.deepcopy(load_states)
    for i, state in enumerate(loaded_states_copy):
        if remove_spikes:
            spectrum = remove_spikes_from_psd(faxis, state['spectrum'])
        else: spectrum = state['spectrum']

        spectrum_dB = dbm_convert(spectrum, offset=offset) - system_gain

        if unit == 'dBm':
            state['spectrum'] = spectrum_dB
        elif unit == 'kelvin':
            spectrum = kelvin_convert(spectrum_dB, df*10**6)
            if normalize is not None:
                state['spectrum'] =  spectrum * normalize
            else:
                state['spectrum'] = spectrum
        else:
            raise ValueError("unit must be dBm or kelvin.")
    return loaded_states_copy