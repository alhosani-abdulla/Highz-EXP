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
LEGEND = ['6" shorted', "8' cable open",'Black body','Ambient temperature load','Noise diode',"8' cable short",'6" open']

if __name__ == "__main__":
    print("Creating image for a specified directory of spectrum files...")
    spec_path = sys.argv[1]
    if spec_path.startswith('~'):
        spec_path = os.path.expanduser(spec_path)
    elif spec_path.startswith('/'):
        pass
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
    plotter.plot_spectrum(dbm_spec_states.values(), save_dir=spec_path, suffix=f'{date_dir}_{os.path.basename(spec_path)}', freq_range=freq_range, ymin=-80, ymax=-30, show_plot=False)
    print(f"Image saved to {spec_path}")
