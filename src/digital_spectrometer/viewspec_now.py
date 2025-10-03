from highz_exp import file_load, spec_plot
import numpy as np
from os.path import join as pjoin
import sys
import os
import glob
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DATA_PATH = '/media/peterson/INDURANCE'

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6
freq_range = (0, 500) # MHz
LEGEND = ['6" shorted', "8' cable open",'Black body','Ambient temperature load','Noise diode',"8' cable short",'6" open']
def start_live_spectrum_view(base_path, ylabel=None, title='Live Spectrum', update_interval=1000):
    """Start a live spectrum view window that automatically finds the latest spectrum file"""
    
    def get_latest_spec_path():
        """Find the most recently created spectrum file in the base directory"""
        try:
            spec_files = []
            for file in os.listdir(base_path):
                if file.endswith('.npy'):
                    full_path = os.path.join(base_path, file)
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
            
            latest_spec = np.load(spec_path, allow_pickle=True).item()
            loaded_spec_states = {os.path.basename(spec_path): latest_spec}
            dbm_spec_states = file_load.preprocess_states(faxis=faxis_hz, load_states=loaded_spec_states, remove_spikes=False, offset=-128, system_gain=0)
            
            ax.clear()
            
            state_name, ntwk = next(iter(dbm_spec_states.items()))
            freq = ntwk.f
            spectrum = np.real(ntwk.s[:, 0, 0])
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
    
if __name__ == "__main__":
    # Find the most recently created directory in DATA_PATH
    directories = glob.glob(pjoin(DATA_PATH, '*/'))
    if directories:
        today_dir = max(directories, key=os.path.getctime)
        print(f"Most recent directory: {today_dir}")
    else:
        print("No directories found in DATA_PATH")
        sys.exit(1)
    
    directories = glob.glob(pjoin(today_dir, '*/'))
    if directories:
        base_path = max(directories, key=os.path.getctime)
        print(f"Most recent sub-directory: {base_path}")
    else:
        print("No sub-directories found in today's directory")
        sys.exit(1)
    
    start_live_spectrum_view(base_path, ylabel='PSD [dBm]', title='Live Spectrum', update_interval=1000)

