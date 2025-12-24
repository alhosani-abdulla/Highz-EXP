import numpy as np
import os, glob, re, pickle, zoneinfo
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
import skrf as rf
import logging

from highz_exp.unit_convert import rfsoc_spec_to_dbm

pjoin = os.path.join
pbase = os.path.basename

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6

class LegacyDSFileLoader():
    """
    A utility class for managing and transforming legacy file format for Digital Spectrometer.

    Sample legacy file output: 
        array({'switch state': 0, 'spectrum': array([274611.,  77320.,  70819., ...,  54413.,  52757.,  48947.])}, dtype=object)

    This loader is specifically designed to handle directories containing 
    fragmented .npy files (each containing 2s-accumulated spectra), providing methods to aggregate them based on 
    temporal metadata encoded in their filenames.

    Attributes:
        dir (str): The source directory path where dataset files are located.
    """
    def __init__(self, dir_path):
        self.dir = dir_path
        pass
    
    def condense_npy_by_timestamp(self, output_dir, pattern='*.npy', time_regex=r'_(\d{6})(?:_|$)', use_pickle=False) -> dict:
        """
        Aggregates multiple .npy files into a single dictionary keyed by timestamp.

        The method performs a five-step pipeline:
        1. Discovery: Finds non-empty files matching the glob pattern.
        2. Extraction: Parses a 'key' (timestamp) from each filename using regex.
        3. Loading: Reads .npy data.
        4. Flattening: Reduces single-item lists to raw objects while keeping collisions as lists.
        5. Serialization: Saves the resulting dictionary to a new file with a generated name.

        Args:
            output_dir (str): Directory where the condensed file will be saved.
            pattern (str): Glob pattern to filter files (default: '*.npy').
            time_regex (str): Regex to extract the time key. Defaults to a 6-digit 'HHMMSS' format.
            use_pickle (bool): If True, saves output as .pkl; otherwise saves as .npy.

        Returns:
            dict | None: The condensed dictionary {timestamp: data} if successful, 
                None if no files were found.

        Raises:
            IOError: If the output directory cannot be created or the file cannot be written.
        """
        dir_path = self.dir
        state_files = self.get_and_clean_nonempty_files(dir_path, pattern)
        raw_buckets = {} # timestamp -> list of data
        first_filenames = {} # timestamp -> first filename found
        
        if not state_files:
            print(f"No files found in {dir_path}")
            return None

        # 2. Load and Group
        for fp in state_files:
            bn = os.path.basename(fp)
            match = re.search(time_regex, bn)
            key = match.group(1) if match else bn.split('_')[1] # fallback
            
            try:
                data = np.load(fp, allow_pickle=True)
                # Use .item() if it's a 0-d array (common for saved dicts), else raw data
                if data.ndim != 0:
                    logging.warning("Possible data corruption detected: npy files containing 2D spectra!")
                    val = data
                else:
                    val = data.item()
                raw_buckets.setdefault(key, []).append(val)
                first_filenames.setdefault(key, bn)
            except Exception as e:
                logging.error(f"Failed to load {fp}: {e}")

        # 3. Flatten
        condensed_flat = {}
        for k,v in raw_buckets.items():
            if len(v) != 1:
                logging.warning(f"Multiple files with the same timestamp {k} detected!")
                condensed_flat[k] = v
            else:
                condensed_flat[k] = v[0]

        # 4. Generate Output Filename
        earliest_key = min(raw_buckets.keys(), key=lambda k: int(k) if k.isdigit() else float('inf'))
        
        d_regex = re.compile(r'(\d{8})')
        p_regex = re.compile(r'(antenna\d*|state\d+|stateOC)', re.IGNORECASE)
        
        base_name = self.get_new_output_name(earliest_key, first_filenames[earliest_key], d_regex, p_regex)
        
        # 5. Save
        os.makedirs(output_dir, exist_ok=True)
        ext = '.pkl' if use_pickle else '.npy'
        output_path = os.path.join(output_dir, f"{base_name}{ext}")

        if use_pickle:
            with open(output_path, 'wb') as f: pickle.dump(condensed_flat, f)
        else:
            np.save(output_path, condensed_flat, allow_pickle=True)

        return condensed_flat

    @staticmethod
    def get_and_clean_nonempty_files(directory, pattern="*.npy") -> list:
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

    @staticmethod
    def load_npy(dir_path, pattern='*state*.npy') -> list:
        """Load all non-empty .npy files from a specified directory and return them as a list.
        
        Return
            - A list of np.array (spectrum)"""
        state_files = LegacyDSFileLoader.get_and_clean_nonempty_files(dir_path, pattern)
        states = []
        for file in state_files:
            states.append(np.load(file, allow_pickle=True).item())
        return states
    
    @staticmethod
    def get_new_output_name(earliest_ts, filename, date_regex, part_regex) -> str:
        """
        Constructs a standardized, descriptive filename ([Date]_[Timestamp]_[Antenna]_[State]) for compressed files by extracting metadata 
        from a sample file path (name).

        It prioritizes the 'earliest' timestamp found in the dataset and attempts 
        to find corresponding identifiers (antenna/state) from the same source filename.

        Args:
            earliest_ts (str): The primary timestamp (e.g., '123456') to include 
                in the filename.
            filename (str): sample filename named in legacy scheme.
            date_regex (re.Pattern): Compiled regex to extract a date string 
                (usually YYYYMMDD).
            part_regex (re.Pattern): Compiled regex to identify 'antenna' or 
                'state' tokens.

        Returns:
            str: A sanitized filename without an extension.
                Example: "20231027_123456_antenna1_state2"

        Note:
            The function ensures only one 'antenna' and one 'state' token are 
            included.
        """
        date_str = None
        parts = []
        
        date_match = date_regex.search(filename)
        if date_match:
            date_str = date_match.group(1)
            
        # Find unique antenna/state tokens
        found_types = set()
        for m in part_regex.finditer(filename):
            token = m.group(1)
            prefix = 'antenna' if 'antenna' in token.lower() else 'state'
            if prefix not in found_types:
                parts.append(token)
                found_types.add(prefix)
                
        name_parts = ([date_str] if date_str else []) + [earliest_ts] + parts
        return re.sub(r'__+', '_', "_".join(name_parts)).strip('_')

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

def get_sorted_time_dirs(date_dir) -> list:
    """
    Returns:
        List[str]: A sorted list of full paths to the subdirectories. 
            Returns an empty list if no directories are found.

    Example:
        >>> get_sorted_time_dirs("/data/2023-10-27")
        ['/data/2023-10-27/1000', '/data/2023-10-27/1100']
    """
    all_items = glob.glob(pjoin(date_dir, "*"))
    time_dirs = [d for d in all_items if os.path.isdir(d)]
    time_dirs.sort()
    logging.info("Found %d time directories in %s", len(time_dirs), date_dir)
    if len(time_dirs) == 0:
        logging.error("No sub directories found in %s", date_dir)
        return
    
    return time_dirs

def get_specs_from_dirs(date_str, time_dirs, state_indx=0) -> dict:
    """Collect all spectrum files for a given date and state index."""
    loaded = {}
    for time_dir in time_dirs:
        loaded.update(add_timestamp(time_dir, date_str, state_indx)) 

    return loaded

def read_loaded(loaded, sort='ascending') -> tuple[np.array, np.array]:
    """Read timestamps and spectra from loaded data. Sort by timestamps.

    Parameters:
    -----------
    loaded: dict. Structure {'timestamp_str': {'spectrum': np.ndarray, 'full_timestamp': datetime, ...}, ...}
    date : str. Formated like 20251216."""
    timestamps = []
    raw_timestamps_str = []
    spectra = []
    for timestamp_str, info_dict in loaded.items():
        timestamps.append(info_dict['full_timestamp'])
        spectrum = rfsoc_spec_to_dbm(info_dict['spectrum'], offset=-128)
        if len(spectrum) != nfft//2:
            logging.warning("Spectrum length %d does not match expected %d for timestamp %s", len(
                spectrum), nfft//2, timestamp_str)
            continue
        spectra.append(spectrum)

    timestamps = np.array(timestamps)
    spectra = np.array(spectra)

    sort_idx = np.argsort(
        timestamps) if sort == 'ascending' else np.argsort(timestamps)[::-1]
    if sort == 'descending':
        sort_idx = sort_idx[::-1]

    return timestamps[sort_idx], spectra[sort_idx]

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