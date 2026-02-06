import numpy as np
import argparse, os, re
from pathlib import Path
import logging, imageio
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional, Union, Dict, Any

from highz_exp import plotter, file_load
from highz_exp.unit_convert import convert_utc_list_to_local
from highz_exp.spec_class import Spectrum
from plot_settings import LEGEND, COLOR_CODE, map_filename_to_legend

pjoin = os.path.join
pbase = os.path.basename

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6

def create_movie_with_imageio(
    image_dir: Union[str, Path], 
    output_filename: str = "spectra_movie.mp4", 
    fps: int = 5
) -> None:
    """
    Creates a movie from plot images using imageio.
    Much simpler than OpenCV as it handles sizing and codecs automatically.
    """
    image_path = Path(image_dir)
    
    # Sort files naturally (batch_1, batch_2, ...)
    files: List[Path] = sorted(
        list(image_path.glob("*batch_*.png")),
        key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x.name)]
    )

    if not files:
        logging.warning("No images found.")
        return

    # Use a 'writer' context manager
    output_path = image_path / output_filename
    with imageio.get_writer(output_path, fps=fps) as writer:
        for filename in files:
            # imageio.imread handles the color space and conversion for you
            image = imageio.imread(filename)
            writer.append_data(image)

    logging.info(f"Movie saved successfully: {output_path}")
    
def create_image_for_condensed(spec_dir: Union[str, Path], state_indx: int = 0, 
    output_dir: Optional[Union[str, Path]] = None, sample: bool = True) -> None:
    """
    Processes spectral data from a directory to generate line plots.
    
    This function loads spectra in condensed format for a specific hardware state, converts timestamps 
    to local time, and generates either a single 'antenna' sample plot or a 
    series of batch plots (10 spectra per image) covering the entire dataset.

    Args:
        spec_dir: Path to the directory containing spectral data files.
        state_indx: State index of the data plotted. int/list of int.
        output_dir: Directory where the plots will be saved. 
            Defaults to spec_dir if None.
        sample: If True, generates only one plot with the first spectrum. 
            If False, generates batch plots of 10 spectra each for the full set.

    Returns:
        None
    """
    spec_path: Path = Path(spec_dir)
    
    # initial metadata derived from directory structure
    initial_time: str = spec_path.name
    initial_date: str = spec_path.parent.name
    
    # Data Loading
    loaded_dict = {}
    if isinstance(state_indx, int): state_indx = [state_indx]
    for i, state in enumerate(state_indx):
        loaded = file_load.get_specs_from_dirs(initial_date, [str(spec_path)], state)
        timestamps, spectra = file_load.read_loaded(loaded)

        if len(spectra) == 0:
            logging.warning(f"No spectra found in {spec_dir} for state {state_indx}")
            return
        # Temporal Processing: Convert UTC to Local
        timestamps: List[datetime] = convert_utc_list_to_local(timestamps, local_timezone=ZoneInfo('HST'))
        loaded_dict[f'state{state}'] = (timestamps, spectra)
        if i == 0:
            # Update date/time strings to reflect local conversion for filenames/titles
            if timestamps:
                first_ts: datetime = timestamps[0]
                date_str: str = first_ts.strftime("%Y-%m-%d")
                time_str: str = first_ts.strftime("%H%M%S")
                logging.info(f"Updated metadata to local time: {date_str} at {time_str}")
            else:
                date_str, time_str = initial_date, initial_time
                logging.warning("No timestamps available; falling back to directory-based metadata.")

    # Directory Management
    final_output_dir: Union[str, Path] = output_dir if output_dir else spec_dir
    os.makedirs(final_output_dir, exist_ok=True)

    # Visualization Configuration
    yticks: List[int] = [-80, -70, -60, -50, -40, -30]
    base_params: Dict[str, Any] = {
        "save_dir": final_output_dir,
        "ylabel": 'PSD [dBm]',
        "ymin": -80,
        "ymax": -30,
        "yticks": yticks,
        "show_plot": False
    }

    if sample:
        ## --- Mode 1: Single Sample Plot ---
        spectrum_list = []
        for state, (timestamps, spectra) in loaded_dict.items():
            sample_spectrum = Spectrum(faxis_hz, spectra[0, :], name=map_filename_to_legend(state))
            spectrum_list.append(sample_spectrum)
        plotter.plot_spectra(spectrum_list, save_path=pjoin(final_output_dir, f'{date_str}_{time_str}_spectra.png'), title=f'{date_str}: {time_str} Spectra',
            **{**base_params, "show_plot": True})
    else:
        # --- Mode 2: Batch Plotting (10 spectra per plot) ---
        for state, (timestamps, spectra) in loaded_dict.items():
            batch_size: int = 10
            total_spectra: int = len(spectra)
            
            for start_idx in range(0, total_spectra, batch_size):
                end_idx: int = min(start_idx + batch_size, total_spectra)
                
                # Prepare Spectrum objects for the current batch
                current_batch: List[Spectrum] = [
                    Spectrum(faxis_hz, spectra[i, :], name=timestamps[i].strftime("%H:%M:%S"))
                    for i in range(start_idx, end_idx)
                ]
                
                batch_num: int = (start_idx // batch_size) + 1
                suffix: str = f'batch_{batch_num:03d}'
                title: str = f'{date_str} {time_str}: Batch {batch_num} ({start_idx}-{end_idx-1})'
                
                plotter.plot_spectra(
                    current_batch, save_path=pjoin(final_output_dir, f'{date_str}_{time_str}_{suffix}.png'),
                    title=title, 
                    **base_params
                )
                
            num_plots = int(np.ceil(total_spectra / batch_size))
            logging.info(f"Generated {num_plots} plots for {total_spectra} spectra.")

            # New Step: Generate the movie
            create_movie_with_imageio(final_output_dir, output_filename=f"spectra_movie_{state}.mp4", fps=2)

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
    yticks = [-80, -70, -60, -50, -40, -30]
    plotter.plot_spectra(dbm_spec_states.values(), save_path=pjoin(spec_path, f'{date_dir}_all_states_spectra.png'), 
                        title=f'{date_dir}: {os.path.basename(spec_path)} Spectra', suffix='all_states',
                        title=f'{date_dir}: {os.path.basename(spec_path)} Spectra', ylabel='PSD [dBm]',
                        ymin=-80, ymax=-30, yticks=yticks, show_plot=show_plots)
    wo_antenna_dbm_states = {k: v for k, v in dbm_spec_states.items() if k != 'Antenna'}

    yticks = [-80, -70, -60, -50, -40, -30]
    plotter.plot_spectra(wo_antenna_dbm_states.values(), 
                         save_path=pjoin(spec_path, f'{date_dir}_wo_antenna_spectra.png'), 
                          title=f'{date_dir}: {os.path.basename(spec_path)} Spectra (w/o Antenna)', ylabel='PSD [dBm]',
                          ymin=-80, ymax=-30, yticks=yticks, show_plot=show_plots)

    print(f"Image saved to {spec_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process spectrum files from a directory.")

    # Required positional argument for input directory
    parser.add_argument("input_dir",  type=str, 
        help="Path to the input directory containing spectrum files")

    # Optional argument for state_indx
    # nargs='+' allows for one or more integers (a list)
    parser.add_argument("--state_indx", type=int, nargs='+', 
        default=[0],
        help="A single state index or a list of indices (e.g., --state_indx 1 2 3)"
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to the output directory where images will be generated.")
    parser.add_argument("--sample", action="store_true",
                        help="Pick only one snapshot of spectra to plot.")

    args = parser.parse_args()

    # Logic from your snippet
    logging.info("Creating image for a specified directory of spectrum files...")
    
    spec_path = os.path.abspath(args.input_dir)
    state_indices = args.state_indx

    print(f"Path: {spec_path}")
    print(f"State Indices: {state_indices}")

    create_image_for_condensed(spec_path, state_indices, output_dir=args.output_dir, sample=args.sample)

    # create_image(spec_path, show_plots=True)
    

