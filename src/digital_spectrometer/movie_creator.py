from highz_exp import file_load, spec_plot
import numpy as np
from os.path import join as pjoin
import sys
import os
from matplotlib.animation import FuncAnimation
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6
freq_range = (0, 500) # MHz
LEGEND = ['6" shorted', "8' cable open",'Black body','Ambient temperature load','Noise diode',"8' cable short",'6" open']

if __name__ == "__main__":
    print("Creating images for all subdirectories containing spectrum files...")
    parent_path = sys.argv[1]
    if parent_path.startswith('~'):
        parent_path = os.path.expanduser(parent_path)
    elif parent_path.startswith('/'):
        pass
    else:
        parent_path = pjoin(os.getcwd(), parent_path)
        print(f"Interpreting path as {parent_path}")
    
    # Find all subdirectories
    subdirs = [d for d in os.listdir(parent_path) if os.path.isdir(pjoin(parent_path, d))]
    
    # If more than 300 subdirs, keep only the 300 most recent ones
    if len(subdirs) > 300:
        # Sort by modification time (most recent first) and take first 300
        subdirs_with_time = [(d, os.path.getmtime(pjoin(parent_path, d))) for d in subdirs]
        subdirs_with_time.sort(key=lambda x: x[1], reverse=True)
        subdirs = [d for d, _ in subdirs_with_time[:300]]
        print(f"Limited to 300 most recent subdirectories out of {len(subdirs_with_time)} total")
    
    print(f"Found {len(subdirs)} subdirectories: {subdirs}")
    
    for subdir in subdirs:
        spec_path = pjoin(parent_path, subdir)
        print(f"Processing {spec_path}...")
        
        try:
            loaded_spec_states = file_load.load_npy_cal(spec_path, pick_snapshot=[1,1,1,1,1,1,1,1,1], cal_names=LEGEND, offset=-128, include_antenna=True)
            dbm_spec_states = file_load.preprocess_states(faxis=faxis_hz, load_states=loaded_spec_states, remove_spikes=False, offset=-128, system_gain=0)
            print("Loaded and preprocessed spectrum states...")
            spec_plot.plot_spectrum(dbm_spec_states, save_dir=spec_path, suffix=subdir, freq_range=freq_range, ymin=-80, ymax=-30, show_plot=False)
            print(f"Image saved to {spec_path}")
        except Exception as e:
            print(f"Error processing {spec_path}: {e}")
    
    print("Finished processing all subdirectories.")
    
    # Collect all PNG files with their creation times
    png_files = []
    for subdir in subdirs:
        png_pattern = pjoin(parent_path, subdir, "*.png")
        subdir_pngs = glob.glob(png_pattern)
        if subdir_pngs:
            # Use directory creation time for ordering
            dir_time = os.path.getmtime(pjoin(parent_path, subdir))
            png_files.extend([(png, dir_time) for png in subdir_pngs])

    # Sort by creation time
    png_files.sort(key=lambda x: x[1])
    png_paths = [png for png, _ in png_files]

    if png_paths:
        print(f"Creating slideshow with {len(png_paths)} images...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        def update_frame(frame):
            ax.clear()
            ax.axis('off')
            img = mpimg.imread(png_paths[frame])
            ax.imshow(img)
            ax.set_title(f"Image {frame+1}/{len(png_paths)}: {os.path.basename(png_paths[frame])}")
            return [ax]
        
        ani = FuncAnimation(fig, update_frame, frames=len(png_paths), 
                           interval=1000, repeat=True, blit=False)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No PNG files found to display.")
    
    