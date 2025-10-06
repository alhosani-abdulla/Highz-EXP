from highz_exp import file_load, spec_plot
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
        
    loaded_spec_states = file_load.load_npy_cal(spec_path, pick_snapshot=[1,1,1,1,1,1,1], cal_names=LEGEND, offset=-128)
    dbm_spec_states = file_load.preprocess_states(faxis=faxis_hz, load_states=loaded_spec_states, remove_spikes=False, offset=-128, system_gain=0)
    print("Loaded and preprocessed spectrum states...")
    spec_plot.plot_spectrum(dbm_spec_states, save_dir=spec_path, suffix='raw', freq_range=freq_range, ymin=-80, ymax=-30, show_plot=False)
    print(f"Image saved to {spec_path}")