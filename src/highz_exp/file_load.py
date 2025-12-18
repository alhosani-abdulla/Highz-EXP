import numpy as np
import os, glob, re, pickle, zoneinfo
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Union, Optional
import skrf as rf
import logging

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

def load_npy(dir_path, pattern='*state*.npy'):
    """Load all non-empty .npy files from a specified directory and return them as a list"""
    state_files = get_and_clean_nonempty_files(dir_path, pattern)
    states = []
    for file in state_files:
        states.append(np.load(file, allow_pickle=True).item())
    return states

def condense_npy_by_timestamp(dir_path, output_dir, pattern='*.npy', time_regex=r'_(\d{6})(?:_|$)', use_pickle=False) -> dict:
    """
    Condense many .npy files in dir_path into a single file keyed by a time-stamp
    extracted from the filename. The returned dictionary is flattened: timestamp -> data
    (or timestamp -> [data, ...] if multiple files share the same timestamp).

    Parameters:
        dir_path (str): directory containing .npy files
        pattern (str): glob pattern for files to include
        time_regex (str): regex with one capture group that extracts the timestamp key from basename. 
            Examples of files matching are: '20210915_123456_antenna1_state2.npy' with regex r'_(\d{6})_' to capture '123456'.
        use_pickle (bool): if True, save with pickle; otherwise save with np.save (allow_pickle=True)

    Returns:
        dict: mapping timestamp -> data or list of data
    """
    state_files = get_and_clean_nonempty_files(dir_path, pattern) # greps all non-empty files matching pattern
    condensed_data = {}      # key -> list of loaded data objects
    filenames_by_key = {}    # key -> list of filenames (used to determine state_name/output filename)

    # fp: file path
    for fp in state_files:
        # bn: base name of files
        bn = pbase(fp)
        # m: match object for time regex
        m = re.search(time_regex, bn)
        if m:
            key = m.group(1)
        else:
            logging.warning(f"This file {fp} does not match the time regex {time_regex}, using alternative key extraction.")
            key = bn.split('_')[1]

        try:
            loaded = np.load(fp, allow_pickle=True)
            try:
                obj = loaded.item()
            except Exception:
                obj = loaded
                logging.warning(f"Loaded object from {fp} is not a dict-like, storing raw loaded object.")
        except Exception as e:
            print(f"Skipping file (load error): {fp} -> {e}")
            continue

        condensed_data.setdefault(key, []).append(obj)
        filenames_by_key.setdefault(key, []).append(bn)

    os.makedirs(output_dir, exist_ok=True)

    if not condensed_data:
        print(f"No files found in {dir_path} matching pattern {pattern}")
        return None

    keys = list(condensed_data.keys())

    def _key_sorter(k):
        return int(k) if k.isdigit() else float('inf')

    earliest_key = min(keys, key=_key_sorter)
    if _key_sorter(earliest_key) == float('inf'):
        earliest_key = min(keys)

    # Determine descriptive parts for output filename from filenames at earliest timestamp
    # Try to extract a date (YYYYMMDD) and antenna/state parts like 'antenna1' and 'state2'
    date_regex = re.compile(r'(\d{8})')
    part_regex = re.compile(r'(antenna\d*|antenna|state\d+|stateOC)', re.IGNORECASE)

    date_str = None
    parts_list = []

    for bn in filenames_by_key.get(earliest_key, []):
        if date_str is None:
            md = date_regex.search(bn)
            if md:
                date_str = md.group(1)

        # collect antenna and state parts in order, avoid duplicates (antenna and state at most once)
        found = {'antenna': False, 'state': False}
        for pm in part_regex.finditer(bn):
            token = pm.group(1)
            tl = token.lower()
            if tl.startswith('antenna') and not found['antenna']:
                parts_list.append(token)
                found['antenna'] = True
            elif tl.startswith('state') and not found['state']:
                parts_list.append(token)
                found['state'] = True
            if found['antenna'] and found['state']:
                break
        if parts_list and date_str:
            break

    # Fallback: if no parsed parts, reuse previous heuristic to get a state-like name
    if not parts_list:
        state_name = None
        pattern_state = re.compile(r'(state\d+|stateOC|antenna\d*|antenna)', re.IGNORECASE)
        for bn in filenames_by_key.get(earliest_key, []):
            m = pattern_state.search(bn)
            if m:
                state_name = m.group(1)
                break
        if state_name is None:
            state_name = os.path.splitext(filenames_by_key[earliest_key][0])[0]
        parts_list = [state_name]

    # Flatten: single-item lists -> the item, multi-item lists kept as lists
    condensed_flat = {}
    for k, lst in condensed_data.items():
        if len(lst) == 1:
            condensed_flat[k] = lst[0]
        else:
            logging.warning(f"Multiple files found for timestamp {k}, storing as list of {len(lst)} items.")
            condensed_flat[k] = lst

    ext = '.pkl' if use_pickle else '.npy'
    if date_str:
        output_basename = f"{date_str}_{earliest_key}_{'_'.join(parts_list)}"
    else:
        output_basename = f"{earliest_key}_{'_'.join(parts_list)}"
    # sanitize any accidental double-underscores
    output_basename = re.sub(r'__+', '_', output_basename).strip('_')
    output_file = os.path.join(output_dir, f"{output_basename}{ext}")

    if use_pickle:
        with open(output_file, 'wb') as fh:
            pickle.dump(condensed_flat, fh)
    else:
        np.save(output_file, condensed_flat, allow_pickle=True)

    print(f"Saved condensed file to: {output_file}")
    return condensed_flat

def load_npy_dict(file_path):
    """Load a .npy file containing a dictionary of timestamped data.

    Parameters:
        file_path (str): Path to the .npy file.

    Returns:
        dict: dictionary of timestamp -> data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}. Double check the path.")

    loaded_dict = np.load(file_path, allow_pickle=True).item()
    if not isinstance(loaded_dict, dict):
        raise ValueError(f"Loaded object from {file_path} is not a dictionary.")
    
    return loaded_dict

def load_npy_cal(dir_path, pick_snapshot=None, cal_names=None, offset=-135, include_antenna=False):
    """Load all calibration state files from a specified directory and return them as a dictionary. 
    
    Parameters:
        pick_snapshot (list, optional): List of indices specifying which snapshot to load for each state.
        cal_names (list, optional): List of calibration state names.
        include_antenna (bool): If True, also load antenna states (state0 and state1).

    Returns:
        dict: Dictionary containing the loaded calibration states: {cal_name: loaded_data}
    """
    from .unit_convert import rfsoc_spec_to_dbm
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
    # state_files.append(get_and_clean_nonempty_files(dir_path, "*stateOC*.npy"))

    load_states = {}

    for i, file_list in enumerate(state_files):
        state_name = cal_names[i] 
        if not file_list:
            continue  # skip if no files found for this state

        if isinstance(pick_snapshot, list):
            idx = pick_snapshot[i]
        elif pick_snapshot is not None:
            spikes = [count_spikes(rfsoc_spec_to_dbm(np.load(file, allow_pickle=True).item()['spectrum'], offset=offset), height=20) for file in file_list]
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

def get_date_state_specs(date_dir, state_indx=0):
    """Collect all spectrum files for a given date and state index."""
    all_items = glob.glob(pjoin(date_dir, "*"))
    time_dirs = [d for d in all_items if os.path.isdir(d)]
    time_dirs.sort()
    logging.info("Found %d time directories in %s", len(time_dirs), date_dir)
    if len(time_dirs) == 0:
        logging.error("No sub directories found in %s", date_dir)
        return
    
    loaded = {}
    for time_dir in time_dirs:
        loaded.update(add_timestamp(time_dir, pbase(date_dir), state_indx)) 

    return loaded

def add_timestamp(time_dir, date_str, state_no) -> dict:
    """Load all spectrum files in a data-collecting cycle (one set of sky spectra + one set of calibration spectra) and return a dict with timestamp keys.

    Parameters:
    -----------
    time_dir : str
                Path to the directory containing spectrum .npy files for a specific cycle.

    Returns:
    --------
    loaded : dict
            Dictionary with the following structure:
            {'timestamp_str': {'spectrum': np.ndarray, ...}, 'full_timestamp': datetime, ...}
    """
    all_specs = sorted(glob.glob(pjoin(time_dir, f"*state{state_no}*")))
    loaded = {}
    for spec_file in all_specs:
        loaded.update(load_npy_dict(spec_file))
    time_dirname = pbase(time_dir)
    datestamp = datetime.strptime(date_str, '%Y%m%d').date()

    for timestamp_str in loaded.keys():
        timestamp = datetime.strptime(timestamp_str, '%H%M%S').time()
        if time_dirname.startswith("23"):
            if timestamp.hour <= 1:
                # Assign to next day
                full_timestamp = datetime.combine(
                    datestamp + timedelta(days=1), timestamp, tzinfo=zoneinfo.ZoneInfo('UTC'))
            else:
                full_timestamp = datetime.combine(
                    datestamp, timestamp, tzinfo=zoneinfo.ZoneInfo('UTC'))
        else:
            full_timestamp = datetime.combine(
                datestamp, timestamp, tzinfo=zoneinfo.ZoneInfo('UTC'))
        loaded[timestamp_str]['full_timestamp'] = full_timestamp
  
    return loaded