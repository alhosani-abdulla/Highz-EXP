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

def create_images_for_date(date_dir):
    """Create and save spectrum images for spectrum files in the specified date directory."""
    spec_path = pjoin(DATA_PATH, date_dir)
    time_dirs = sorted(
        [d for d in os.listdir(spec_path) if os.path.isdir(pjoin(spec_path, d))],
        key=lambda d: os.path.getmtime(pjoin(spec_path, d)),
        reverse=True
    )
    if not time_dirs:
        print(f"No data found in {spec_path}.")
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = []
        for time_dir in time_dirs:
            full_spec_path = pjoin(spec_path, time_dir)
            futures.append(executor.submit(create_image, full_spec_path, show_plots=False))
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
    
    # If more than  subdirs, keep only the 50 most recent ones
    if len(subdirs) > 50:
        # Sort by modification time (most recent first) and take first 50
        subdirs = subdirs[:50]
        print(f"Limited to 50 most recent subdirectories out of {len(subdirs)} total")
    else:
        print(f"Found {len(subdirs)} subdirectories: {subdirs}")
    
    for date_dir in subdirs:
        print(f"Processing date directory: {date_dir}")
        create_images_for_date(date_dir)
    
    # Collect all PNG files with their creation times
    png_files = []
    for subdir in subdirs:
        png_pattern = pjoin(parent_path, subdir, "*.png")
        subdir_pngs = glob.glob(png_pattern)
        if subdir_pngs:
            # Use directory creation time for ordering
            dir_time = os.path.getmtime(pjoin(parent_path, subdir))
            png_files.extend([(png, dir_time) for png in subdir_pngs])

    # # Sort by creation time
    # png_files.sort(key=lambda x: x[1])
    # png_paths = [png for png, _ in png_files]

    # if png_paths:
    #     print(f"Creating slideshow with {len(png_paths)} images...")
        
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     ax.axis('off')
        
    #     def update_frame(frame):
    #         ax.clear()
    #         ax.axis('off')
    #         img = mpimg.imread(png_paths[frame])
    #         ax.imshow(img)
    #         ax.set_title(f"Image {frame+1}/{len(png_paths)}: {os.path.basename(png_paths[frame])}")
    #         return [ax]
        
    #     ani = FuncAnimation(fig, update_frame, frames=len(png_paths), 
    #                        interval=1000, repeat=True, blit=False)
        
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("No PNG files found to display.")
    
    