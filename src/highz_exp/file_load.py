import numpy as np
import os, glob, copy, re, pickle
import skrf as rf

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

def count_spikes(y, x=None, height=None, threshold=None, distance=None, print_table=False) -> np.ndarray:
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
        spike_data (np.ndarray): 2D array with shape (n_spikes, 2) containing [x_vals, heights].
    """
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(y, height=height, threshold=threshold, distance=distance)
    heights = properties.get('peak_heights', np.array([]))
    
    if x is None:
        x_vals = peaks
    else:
        x_vals = np.asarray(x)[peaks]
    
    # Create 2D array with x_vals and heights
    if len(x_vals) > 0:
        spike_data = np.column_stack((x_vals, heights))
    else:
        spike_data = np.empty((0, 2))
    
    if print_table:
        print("Spike # |    x    |  height")
        print("---------------------------")
        for i, (xi, hi) in enumerate(zip(x_vals, heights)):
            print(f"{i+1:7d} | {xi:7.3f} | {hi:7.3f}")
    
    return spike_data

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
    state_files = get_and_clean_nonempty_files(dir_path, pattern)
    states = []
    for file in state_files:
        states.append(np.load(file, allow_pickle=True).item())
    return states

def condense_npy_by_timestamp(dir_path, output_dir, pattern='*.npy', time_regex=r'_(\d{6})(?:_|$)', use_pickle=False):
    """
    Condense many .npy files in dir_path into a single file keyed by a time-stamp
    extracted from the filename. The returned dictionary is flattened: timestamp -> data
    (or timestamp -> [data, ...] if multiple files share the same timestamp).

    Parameters:
        dir_path (str): directory containing .npy files
        pattern (str): glob pattern for files to include
        time_regex (str): regex with one capture group that extracts the timestamp key from basename
        use_pickle (bool): if True, save with pickle; otherwise save with np.save (allow_pickle=True)

    Returns:
        dict: mapping timestamp -> data or list of data
    """
    state_files = get_and_clean_nonempty_files(dir_path, pattern)
    condensed_data = {}      # key -> list of loaded data objects
    filenames_by_key = {}    # key -> list of filenames (used to determine state_name/output filename)

    for fp in state_files:
        bn = pbase(fp)
        m = re.search(time_regex, bn)
        if m:
            key = m.group(1)
        else:
            key = os.path.splitext(bn)[0]

        try:
            loaded = np.load(fp, allow_pickle=True)
            try:
                obj = loaded.item()
            except Exception:
                obj = loaded
        except Exception as e:
            print(f"Skipping file (load error): {fp} -> {e}")
            continue

        condensed_data.setdefault(key, []).append(obj)
        filenames_by_key.setdefault(key, []).append(bn)

    os.makedirs(output_dir, exist_ok=True)

    if not condensed_data:
        raise ValueError(f"No files found in {dir_path} matching pattern {pattern}")

    keys = list(condensed_data.keys())

    def _key_sorter(k):
        return int(k) if k.isdigit() else float('inf')

    earliest_key = min(keys, key=_key_sorter)
    if _key_sorter(earliest_key) == float('inf'):
        earliest_key = min(keys)

    # Determine state name for output file from filenames at earliest timestamp
    state_name = None
    for bn in filenames_by_key.get(earliest_key, []):
        m = re.search(r'(state\d+|stateOC)', bn, re.IGNORECASE)
        if m:
            state_name = m.group(1)
            break
    if state_name is None:
        state_name = os.path.splitext(filenames_by_key[earliest_key][0])[0]

    # Flatten: single-item lists -> the item, multi-item lists kept as lists
    condensed_flat = {}
    for k, lst in condensed_data.items():
        if len(lst) == 1:
            condensed_flat[k] = lst[0]
        else:
            condensed_flat[k] = lst

    ext = '.pkl' if use_pickle else '.npy'
    output_file = os.path.join(output_dir, f"{earliest_key}_{state_name}{ext}")
    if use_pickle:
        with open(output_file, 'wb') as fh:
            pickle.dump(condensed_flat, fh)
    else:
        np.save(output_file, condensed_flat, allow_pickle=True)

    print(f"Saved condensed file to: {output_file}")
    return condensed_flat

def load_npy_cal(dir_path, pick_snapshot=None, cal_names=None, offset=-135, include_antenna=False):
    """Load all calibration state files from a specified directory and return them as a dictionary. 
    
    Parameters:
        pick_snapshot (list, optional): List of indices specifying which snapshot to load for each state.
        cal_names (list, optional): List of calibration state names.
        include_antenna (bool): If True, also load antenna states (state0 and state1).

    Returns:
        dict: Dictionary containing the loaded calibration states.
    """
    state_files = []
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory does not exist: {dir_path}. Double check the path.")

    if not get_and_clean_nonempty_files(dir_path, '*state*.npy'):
        raise FileNotFoundError(f"No '*state*.npy' files found in directory: {dir_path}")
    
    # Load antenna states if requested
    if include_antenna:
        state_files.append(get_and_clean_nonempty_files(dir_path, f'*state0*.npy'))
        state_files.append(get_and_clean_nonempty_files(dir_path, f'*state1*.npy'))
    else:
        _ = get_and_clean_nonempty_files(dir_path, f'*state1*.npy')
        _ = get_and_clean_nonempty_files(dir_path, f'*state0*.npy')
    
    # Load calibration states (state2-7)
    for i in range(2, 8):
        state_files.append(get_and_clean_nonempty_files(dir_path, f'*state{i}*.npy'))
    state_files.append(get_and_clean_nonempty_files(dir_path, "*stateOC*.npy"))

    load_states = {}
    def get_state_name(i):
        if include_antenna:
            if i == 0:
                return 'antenna (powered)'
            elif i == 1:
                return 'antenna (unpowered)'
            else:
                if cal_names is not None:
                    return cal_names[i-2]
                else:
                    return f'state{i}'
        else:
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
    
    load_type = "calibration and antenna states" if include_antenna else "calibration states"
    print(f"Loading {load_type} measurements in digital spectrometer box, in the recorded power with no conversion.")

    return load_states

def states_to_ntwk(f, loaded_states):
    """Convert loaded spectrum states to a dictionary of rf.Network objects with frequency f."""
    if isinstance(loaded_states, dict):
        ntwk_dict = {}
        for state_name, state in loaded_states.items():
            spectrum = state['spectrum']
            ntwk_dict[state_name] = rf.Network(f=f, name=state_name, s=spectrum.reshape(-1, 1, 1))
        print("Returning networks of (raw) recorded spectra.")
        return ntwk_dict

def preprocess_states(faxis, load_states, remove_spikes=True, unit='dBm', offset=-135, system_gain=100, normalize=None) -> dict:
    """Preprocess the loaded states by converting the spectrum to the specified unit and removing spikes if required. 
    
    Parameters:
        faxis: np.ndarray, frequency points in MHz.
        system_gain: float, the system gain in dB to be discounted from the recorded spectrum.

    Returns:
        dict: A dictionary of processed network objects.
    """
    df = float(faxis[1] - faxis[0])
    loaded_states_copy = copy.deepcopy(load_states)
    ntwk_dict = {}
    for label, state in loaded_states_copy.items():
        if remove_spikes:
            spectrum = remove_spikes_from_psd(faxis, state['spectrum'])
        else: spectrum = state['spectrum']

        spectrum_dBm = spec_to_dbm(spectrum, offset=offset) - system_gain

        if unit == 'dBm':
            state['spectrum'] = spectrum_dBm
        elif unit == 'kelvin':
            spectrum = dbm_to_kelvin(spectrum_dBm, df)
            if normalize is not None:
                state['spectrum'] =  spectrum * normalize
            else:
                state['spectrum'] = spectrum
        else:
            raise ValueError("unit must be dBm or kelvin.")
    
    for label, state in loaded_states_copy.items():
        spectrum = state['spectrum']
        ntwk_dict[label] = rf.Network(f=faxis, name=label, s=spectrum.reshape(-1, 1, 1))

    return ntwk_dict
    
def norm_states(f, loaded_states, ref_state_label, ref_temp=300, system_gain=100) -> tuple:
    """Normalize loaded RAW! spectra from digital spectrometer to a reference state and convert to Kelvin.

    Returns:
    loaded_states_kelvin: dict
        Dictionary of loaded states with spectra converted to Kelvin.
    gain: np.ndarray
        Normalization factor applied to convert from dBm to Kelvin.
    """
    dbm = np.array(spec_to_dbm(remove_spikes_from_psd(f, loaded_states[ref_state_label]['spectrum'])))-system_gain
    gain = norm_factor(dbm, ref_temp)
    loaded_states_kelvin = preprocess_states(f, loaded_states, unit='kelvin', normalize=gain, system_gain=system_gain)
    return loaded_states_kelvin, gain