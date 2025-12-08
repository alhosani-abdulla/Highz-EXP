from highz_exp import file_load
import numpy as np
import sys, os, glob
from os.path import join as pjoin
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from highz_exp.spec_class import Spectrum

DATA_PATH = '/home/peterson/Data/INDURANCE'

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6
freq_range = (0, 500) # MHz
LEGEND = ['6" shorted', "8' cable open",'Black body','Ambient temperature load','Noise diode',"8' cable short",'6" open']

def map_filename_to_legend(statename):
    mapping = {'state0': 'Antenna (Power on)', 'state1': 'Antenna (Power off)',
               'state2': '6" shorted', 'state3': "8' cable open",
               'state4': 'Black body', 'state5': 'Ambient temperature load',
               'state6': 'Noise diode', 'state7': "8' cable short",
               'stateOC': '6" open'}
    return mapping.get(statename, statename)

def parse_filename(spec_path) -> tuple[str, str, str]:
    """Parse the spectrum file name to extract state_name, antenna_name, and time_stamp"""
    pbase = os.path.basename
    filename = pbase(spec_path).split('.')[0]
    spec_state = filename.split('_')[-1]
    antenna_name = filename.split('_')[-2]
    time_stamp = filename.split('_')[-3]

    return spec_state, antenna_name, time_stamp

# Modify the start_live_spectrum_view function to use dynamic base path
def start_live_spectrum_view_dynamic(ylabel=None, update_interval=1000):
    """Start a live spectrum view window that automatically finds the latest spectrum file"""
    
    def get_latest_spec_path():
        """Find the most recently created spectrum file in the latest base directory"""
        try:
            current_base_path = get_latest_base_path()
            if current_base_path is None:
                return None
            
            spec_files = []
            for file in os.listdir(current_base_path):
                if file.endswith('.npy'):
                    full_path = os.path.join(current_base_path, file)
                    spec_files.append((full_path, os.path.getctime(full_path)))
            
            if not spec_files:
                return None
            
            latest_file = max(spec_files, key=lambda x: x[1])
            return latest_file[0]
        except Exception as e:
            print(f"Error finding latest spec file: {e}")
            return None
    
    def update_plot(frame):
        """Update the plot with the latest data"""
        try:
            spec_path = get_latest_spec_path()
            if spec_path is None:
                return
            
            pbase = os.path.basename
            
            latest_spec = np.load(spec_path, allow_pickle=True).item()
            time_dir = pbase(os.path.dirname(spec_path))
            date_dir = pbase(os.path.dirname(time_dir))

            spec_state, antenna_name, time_stamp = parse_filename(spec_path)
            spec_name = map_filename_to_legend(spec_state)

            spectrum = Spectrum(faxis_hz, latest_spec['spectrum'], 
                                name=spec_name)
    
            title = f'Live Spectrum - {antenna_name} - {date_dir}'
            state_name = f'{time_stamp}: {spec_name}'
            loaded_spec_states = {state_name: latest_spec}

            dbm_spec_states = file_load.preprocess_states(load_states=loaded_spec_states, remove_spikes=False, offset=-128, system_gain=0)
            
            ax.clear()
            
            state_name, spec = next(iter(dbm_spec_states.items()))
            freq = spec.f
            spectrum = spec.s
            faxis_mhz = freq / 1e6
            
            ax.plot(faxis_mhz, spectrum, label=state_name)
            ax.set_ylim(-80, -30)
            ax.set_xlim(*freq_range)
            ax.legend(fontsize=18)
            ax.set_ylabel(ylabel if ylabel else 'PSD [dBm]', fontsize=20)
            ax.set_xlabel('Frequency [MHz]', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.set_title(title, fontsize=22)
            ax.grid(True)
            
            fig.tight_layout()
            
        except Exception as e:
            print(f"Error updating plot: {e}")
    
    def on_closing():
        """Handle window closing"""
        root.quit()
        root.destroy()
    
    try:
        # Create window and plot
        root = tk.Tk()
        root.title("Live Spectrum Viewer")
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        ani = animation.FuncAnimation(fig, update_plot, interval=update_interval, blit=False)
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        if 'root' in locals():
            root.quit()
            root.destroy()
    except Exception as e:
        print(f"Unexpected error: {e}")
        if 'root' in locals():
            root.quit()
            root.destroy()

def get_latest_base_path() -> str:
    """Find the most recently created subdirectory in the most recent directory"""
    try:
        # Find the most recently created directory in DATA_PATH
        directories = glob.glob(pjoin(DATA_PATH, '*/'))
        if not directories:
            return None
        
        today_dir = max(directories, key=os.path.getctime)
        
        # Find the most recently created subdirectory
        subdirectories = glob.glob(pjoin(today_dir, '*/'))
        if not subdirectories:
            return None
        
        base_path = max(subdirectories, key=os.path.getctime)
        return base_path
    except Exception as e:
        print(f"Error finding latest base path: {e}")
        return None

if __name__ == "__main__":

    
    # Initial path check
    initial_base_path = get_latest_base_path()
    if initial_base_path is None:
        print("No directories found in DATA_PATH")
        sys.exit(1)
    
    print(f"Starting with base path: {initial_base_path}")
    
    start_live_spectrum_view_dynamic(ylabel='PSD [dBm]', update_interval=1000)

