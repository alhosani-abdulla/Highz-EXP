from highz_exp import plotter, file_load
from highz_exp.spec_class import Spectrum
import numpy as np
from os.path import join as pjoin
import sys
import os

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6
freq_range = (0, 500) # MHz
mapping = {'state0': 'Antenna', 'state1': 'Open Circuit',
    'state2': 'Short', 'state3': "Long cable short",
    'state4': 'Black body', 'state5': 'Ambient temperature load',
    'state6': 'Noise diode', 'state7': "Long cable open",
    'stateOC': '6" open'}
LEGEND = ['Antenna', 'Open Circuit', 'Short', 'Long cable short',
          'Black body', 'Ambient temperature load', 'Noise diode', 'Long cable open']

if __name__ == "__main__":
    print("Creating image for a specified directory of spectrum files...")
    spec_path = sys.argv[1]
    if spec_path.startswith('~'):
        spec_path = os.path.expanduser(spec_path)
    elif spec_path.startswith('/'):
        pass
    elif spec_path.startswith('.'):
        spec_path = os.path.abspath(spec_path)
    else:
        spec_path = pjoin(os.getcwd(), spec_path)
        print(f"Interpreting path as {spec_path}")
    
    loaded_spec_npys = file_load.load_npy_cal(spec_path, pick_snapshot=[1,1,1,1,1,1,1,1,1], cal_names=LEGEND, offset=-128, include_antenna=True)
    spectrum_dicts = {}
    for spec_name, latest_npy_load in loaded_spec_npys.items():
        spectrum = Spectrum(faxis_hz, latest_npy_load['spectrum'], name=spec_name)
        spectrum_dicts[spec_name] = spectrum
    dbm_spec_states = Spectrum.preprocess_states(load_states=spectrum_dicts, remove_spikes=False, offset=-128, system_gain=0)
    print("Loaded and preprocessed spectrum states...")
    date_dir = os.path.basename(os.path.dirname(spec_path))
    yticks = [-80, -70, -60, -50, -40, -30, -20]
    plotter.plot_spectrum(dbm_spec_states.values(), save_dir=spec_path, suffix='all_states',
                          title=f'{date_dir}: {os.path.basename(spec_path)} Spectra', ylabel='PSD [dBm]',
                          ymin=-80, ymax=-20, yticks=yticks, show_plot=True)
    print(f"Image saved to {spec_path}")
