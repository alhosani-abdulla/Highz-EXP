import numpy as np
from os.path import join as pjoin
import concurrent.futures
import os, sys, glob
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from highz_exp import file_load, plotter
from image_creator import create_image
from plot_settings import LEGEND, DATA_PATH, COLOR_CODE

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6
freq_range = (0, 500) # MHz

def create_image_if_not_found(spec_path):
    """Create and save spectrum images for all spectrum files in the specified directory if not already present."""
    print("Processing time directory:", spec_path)
    png_files = glob.glob(pjoin(spec_path, "*.png"))
    pbase = os.path.basename
    png_names = [pbase(png) for png in png_files]
    if "spectrum_all_states.png" in png_names and "spectrum_wo_antenna.png" in png_names:
        print(f"Images already exist in {spec_path}, skipping image creation.")
        return
    create_image(spec_path, show_plots=False)

def create_images_for_date(date_dir):
    """Create and save spectrum images for spectrum files in the specified date directory."""
    spec_path = pjoin(DATA_PATH, date_dir)
    time_dirs = sorted(
        [d for d in os.listdir(spec_path) if os.path.isdir(pjoin(spec_path, d))],
        key=lambda d: os.path.getmtime(pjoin(spec_path, d)),
        reverse=True
    )[:20]
    if not time_dirs:
        print(f"No data found in {spec_path}.")
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for time_dir in time_dirs:
            full_spec_path = pjoin(spec_path, time_dir)
            futures.append(executor.submit(create_image_if_not_found, full_spec_path))
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing a spectrum directory: {e}")

def find_images(date_dir):
    """Find all PNG images in the specified date directory."""
    spec_path = pjoin(DATA_PATH, date_dir)
    png_files = glob.glob(pjoin(spec_path, "*.png"))
    return png_files

if __name__ == "__main__":
    print("Creating images for all subdirectories containing spectrum files...")
    parent_path = DATA_PATH
    if parent_path.startswith('~'):
        parent_path = os.path.expanduser(parent_path)
    elif parent_path.startswith('/'):
        pass
    else:
        parent_path = pjoin(os.getcwd(), parent_path)
        print(f"Interpreting path as {parent_path}")
    
    # Get all subdirectories sorted by modification time (most recent first)
    # Keep only the 2 most recent ones for processing
    subdirs = sorted(
        [d for d in os.listdir(parent_path) if os.path.isdir(pjoin(parent_path, d)) and d != 'logs'],
        key=lambda d: os.path.getmtime(pjoin(parent_path, d)),
        reverse=True
    )[:2]
    
    print("Two most recent dates of data to process:", subdirs)
    for date_dir in subdirs:
        print(f"Processing date directory: {date_dir}")
        create_images_for_date(date_dir)
        
        # Find and display the 20 most recent images after creation
        png_files = glob.glob(pjoin(parent_path, date_dir, "**", "*.png"), recursive=True)
        png_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        png_files = png_files[:20]  # Keep only the 20 most recent
        
        if png_files:
            print(f"Displaying {len(png_files)} most recent images from {date_dir}...")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            
            def update_frame(frame):
                ax.clear()
                ax.axis('off')
                img = mpimg.imread(png_files[frame])
                ax.imshow(img)
                ax.set_title(f"Image {frame+1}/{len(png_files)}")
                return [ax]
                
            ani = FuncAnimation(fig, update_frame, frames=len(png_files), 
                    interval=1000, repeat=True, blit=False)
            
            plt.tight_layout()
            plt.show()

    
    