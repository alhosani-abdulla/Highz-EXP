from matplotlib.widgets import Slider
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import os, time, glob, threading, heapq
from scipy.signal import find_peaks

def dbm_convert(spectrum):
    """Convert SPD into dBm scale."""
    return [10 * np.log10(s) - 135 for s in spectrum]

def load_npy_spec(data_full_path, pick_snapshot=None):
  state_files = []
  _ = get_and_clean_nonempty_files(data_full_path, f'*state1*.npy')
  _ = get_and_clean_nonempty_files(data_full_path, f'*state0*.npy')
  for i in range(2, 8):
    state_files.append(get_and_clean_nonempty_files(data_full_path, f'*state{i}*.npy'))
  state_files.append(get_and_clean_nonempty_files(data_full_path, "*stateOC*.npy"))

  load_states = []
  for i, file_list in enumerate(state_files):
    if pick_snapshot is not None:
      spikes = [count_spikes(dbm_convert(np.load(file, allow_pickle=True).item()['spectrum']), height=20) for file in file_list]
      if min(spikes) > 0:
        print(f"All the recorded spectrum files for state{i+2} have spikes with more than 20 dB of height")
      indx_least_spikes = np.argmin(spikes)
      indx_least_spikes = -1
      load_states.append(np.load(file_list[indx_least_spikes], allow_pickle=True).item())
    elif isinstance(pick_snapshot, list):
      load_states.append(np.load(file_list[pick_snapshot[i]], allow_pickle=True).item())
    else:
      load_states.append(np.load(file_list[0], allow_pickle=True).item())

  return load_states

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
    peaks, _ = find_peaks(y, height=height, threshold=threshold, distance=distance)
    return len(peaks)