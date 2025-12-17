import numpy as np
from os.path import join as pjoin
import sys
import os

from highz_exp import plotter, file_load
from highz_exp.spec_class import Spectrum
from plot_settings import LEGEND, COLOR_CODE, LEGEND_WO_ANTENNA


nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6

def create_image(spec_path, show_plots=False):
    """Create and save spectrum images for all spectrum files (wo antenna vs. with antenna) in the specified directory."""
    loaded_spec_npys = file_load.load_npy_cal(spec_path, pick_snapshot=[1,1,1,1,1,1,1,1,1], cal_names=LEGEND, offset=-128, include_antenna=True)
    spectrum_dicts = {}
    for spec_name, latest_npy_load in loaded_spec_npys.items():
        spectrum = Spectrum(faxis_hz, latest_npy_load['spectrum'], name=spec_name, colorcode=COLOR_CODE.get(spec_name, None))
        spectrum_dicts[spec_name] = spectrum
    dbm_spec_states = Spectrum.preprocess_states(load_states=spectrum_dicts, remove_spikes=False, offset=-128, system_gain=0)
    print("Loaded and preprocessed spectrum states...")
    date_dir = os.path.basename(os.path.dirname(spec_path))
    yticks = [-80, -70, -60, -50, -40, -30, -20]
    plotter.plot_spectrum(dbm_spec_states.values(), save_dir=spec_path, suffix='all_states',
                          title=f'{date_dir}: {os.path.basename(spec_path)} Spectra', ylabel='PSD [dBm]',
                          ymin=-80, ymax=-20, yticks=yticks, show_plot=show_plots)
    wo_antenna_dbm_states = {k: v for k, v in dbm_spec_states.items() if k != 'Antenna'}

    yticks = [-80, -70, -60, -50, -40, -30]
    plotter.plot_spectrum(wo_antenna_dbm_states.values(), save_dir=spec_path, suffix='wo_antenna',
                          title=f'{date_dir}: {os.path.basename(spec_path)} Spectra (w/o Antenna)', ylabel='PSD [dBm]',
                          ymin=-80, ymax=-30, yticks=yticks, show_plot=show_plots)

    print(f"Image saved to {spec_path}")

if __name__ == "__main__":
    print("Creating image for a specified directory of spectrum files...")
    spec_path = sys.argv[1]
    spec_path = os.path.abspath(spec_path)
    # if spec_path.startswith('~'):
    #     spec_path = os.path.expanduser(spec_path)
    # elif spec_path.startswith('/'):
    #     pass
    # elif spec_path.startswith('.'):
    #     spec_path = os.path.abspath(spec_path)
    # else:
    #     spec_path = pjoin(os.getcwd(), spec_path)
    #     print(f"Interpreting path as {spec_path}")
    create_image(spec_path, show_plots=True)
    

