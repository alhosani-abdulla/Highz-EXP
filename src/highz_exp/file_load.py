import numpy as np
import os, glob, copy
import skrf as rf
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from scipy.constants import Boltzmann as k_B
from .unit_convert import spec_to_dbm, dbm_to_kelvin, norm_factor

pjoin = os.path.join
pbase = os.path.basename

def load_s1p(s1p_files, labels=None) -> dict:
    """
    Load S1P files and return a dictionary of label: rf.Network objects.

    Parameters:
    - s1p_files (list of str): Paths to .s1p files.
    - labels (list of str, optional): Labels for the files.

    Returns:
    - dict: {label: rf.Network}
    """
    if labels is None:
        labels = [pbase(f) for f in s1p_files]

    return {label: rf.Network(file) for file, label in zip(s1p_files, labels)}

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

def count_spikes(y, x=None, height=None, threshold=None, distance=None, print_table=False):
    """
    Count the number of peaks in a signal that exceed a specified height and threshold.
    Optionally return the heights and print a table of (x, height).

    Parameters:
        y (array-like): The signal data.
        x (array-like, optional): The x-axis values corresponding to y. If None, indices are used.
        height (float or None): Required height of peaks.
        threshold (float or None): Required threshold of peaks.
        distance (int or None): Required minimal horizontal distance (in samples) between neighboring peaks.
        print_table (bool): If True, print a table of (x, height) for each detected spike.

    Returns:
        count (int): Number of detected spikes.
        heights (np.ndarray): Heights of the detected spikes.
    """
    peaks, properties = find_peaks(y, height=height, threshold=threshold, distance=distance)
    heights = properties.get('peak_heights', np.array([]))
    if print_table:
        if x is None:
            x_vals = peaks
        else:
            x_vals = np.asarray(x)[peaks]
        print("Spike # |    x    |  height")
        print("---------------------------")
        for i, (xi, hi) in enumerate(zip(x_vals, heights)):
            print(f"{i+1:7d} | {xi:7.3f} | {hi:7.3f}")

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

def load_npy_cal(dir_path, pick_snapshot=None, cal_names=None, offset=-135):
    """Load all calibration state files from a specified directory and return them as a dictionary.

    Parameters:
        pick_snapshot (list, optional): List of indices specifying which snapshot to load for each state.
        cal_names (list, optional): List of calibration state names.

    Returns:
        dict: Dictionary containing the loaded calibration states.
    """
    data_full_path = pjoin(dir_path)
    state_files = []
    _ = get_and_clean_nonempty_files(data_full_path, f'*state1*.npy')
    _ = get_and_clean_nonempty_files(data_full_path, f'*state0*.npy')
    for i in range(2, 8):
        state_files.append(get_and_clean_nonempty_files(data_full_path, f'*state{i}*.npy'))
    state_files.append(get_and_clean_nonempty_files(data_full_path, "*stateOC*.npy"))

    load_states = {}
    def get_state_name(i):
        if cal_names is not None:
            return cal_names[i]
        else:
            return f'state{i+2}'

    for i, file_list in enumerate(state_files):
        state_name = get_state_name(i)
        if not file_list:
            continue  # skip if no files found for this state

        if isinstance(pick_snapshot, list):
            idx = pick_snapshot[i]
        elif pick_snapshot is not None:
            spikes = [count_spikes(spec_to_dbm(np.load(file, allow_pickle=True).item()['spectrum'], offset=offset), height=20) for file in file_list]
            if min(spikes) > 0:
                print(f"All the recorded spectrum files for {state_name} have spikes with more than 20 dB of height")
            idx = int(np.argmin(spikes))
        else:
            idx = 0

        load_states[state_name] = np.load(file_list[idx], allow_pickle=True).item()
    
    print("Loading calibration states measurements in digital spectrometer box.")

    return load_states

def states_to_ntwk(f, loaded_states, offset=-135):
    """Convert loaded spectrum states to a dictionary of rf.Network objects with frequency f."""
    if isinstance(loaded_states, dict):
        ntwk_dict = {}
        for state_name, state in loaded_states.items():
            spectrum = spec_to_dbm(state['spectrum'], offset=offset)
            ntwk_dict[state_name] = rf.Network(f=f, name=state_name, s=spectrum.reshape(-1, 1, 1))
        return ntwk_dict

def preprocess_states(faxis, load_states, remove_spikes=True, unit='dBm', offset=-135, system_gain=100, normalize=None):
    """Preprocess the loaded states by converting the spectrum to the specified unit and removing spikes if required
    
    Parameters:
        faxis: np.ndarray, frequency points in MHz. """
    df = float(faxis[1] - faxis[0])
    loaded_states_copy = copy.deepcopy(load_states)
    for i, state in enumerate(loaded_states_copy):
        if remove_spikes:
            spectrum = remove_spikes_from_psd(faxis, state['spectrum'])
        else: spectrum = state['spectrum']

        spectrum_dBm = spec_to_dbm(spectrum, offset=offset) - system_gain

        if unit == 'dBm':
            state['spectrum'] = spectrum_dBm
        elif unit == 'kelvin':
            spectrum = dbm_to_kelvin(spectrum_dBm, df * 10**6)
            if normalize is not None:
                state['spectrum'] =  spectrum * normalize
            else:
                state['spectrum'] = spectrum
        else:
            raise ValueError("unit must be dBm or kelvin.")
    return loaded_states_copy

def norm_states(f, loaded_states, ref_state_indx=3, ref_temp=300, system_gain=100):
    """Normalize loaded states to a reference state and convert to Kelvin.

    Parameters:
        f: np.ndarray. Frequency points in MHz.
    
    Return
    loaded_states_kelvin: dict
        Dictionary of loaded states with spectra converted to Kelvin.
    """
    dbm = np.array(spec_to_dbm(remove_spikes_from_psd(f, loaded_states[ref_state_indx]['spectrum'])))-system_gain
    gain = norm_factor(dbm, ref_temp)
    loaded_states_kelvin = preprocess_states(f, loaded_states, unit='kelvin', normalize=gain, system_gain=system_gain)
    return loaded_states_kelvin