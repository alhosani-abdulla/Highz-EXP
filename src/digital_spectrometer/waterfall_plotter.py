# Adapted from Marcus's waterfall plotter for digital spectrometer data
import os, glob, logging, sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from highz_exp.unit_convert import rfsoc_spec_to_dbm
from highz_exp.file_load import load_npy_dict
from datetime import datetime, date

nfft = 32768
fs = 3276.8/4
fbins = np.arange(0, nfft//2)
df = fs/nfft
faxis = fbins*df
faxis_hz = faxis*1e6

pjoin = os.path.join

def get_date_state_specs(date_dir, state_indx=0):
	"""Collect all spectrum files for a given date and state index."""
	time_dirs = sorted(glob.glob(pjoin(date_dir, "*")))
	logging.info("Found %d time directories in %s", len(time_dirs), date_dir)
	if len(time_dirs) == 0:
		logging.error("No sub directories found in %s", date_dir)
		return
	
	all_specs = [f for time_dir in time_dirs for f in sorted(glob.glob(pjoin(time_dir, "*")))]
	all_state_specs = [f for f in all_specs if f"state{state_indx}" in f]
	logging.info("Found %d spectra for state %d", len(all_state_specs), state_indx)
 
	loaded = {}
	for spec_file in all_state_specs:
		loaded.update(load_npy_dict(spec_file))
	return loaded

def read_loaded(loaded, sort='ascending') -> tuple[np.array, np.array]:
	"""Read timestamps and spectra from loaded data. Sort by timestamps."""
	timestamps = []
	spectra = []
	for timestamp_str, spec_dict in loaded.items():
		timestamps.append(datetime.strptime(timestamp_str, '%H%M%S').time())
		spectrum = rfsoc_spec_to_dbm(spec_dict['spectrum'], offset=-128)
		if len(spectrum) != nfft//2:
			logging.warning("Spectrum length %d does not match expected %d for timestamp %s", len(spectrum), nfft//2, timestamp_str)
			continue
		spectra.append(spectrum)
	
	timestamps = np.array(timestamps)
	spectra = np.array(spectra)
	
	sort_idx = np.argsort(timestamps) if sort == 'ascending' else np.argsort(timestamps)[::-1]
	if sort == 'descending':
		sort_idx = sort_idx[::-1]
	
	return timestamps[sort_idx], spectra[sort_idx]

def plot_waterfall_heatmap(timestamps, spectra, faxis_mhz, title, output_path=None, show_plot=True):
	"""Create a heatmap of spectra with power levels as color coding."""
	plt.figure(figsize=(12, 6))
	
	# Convert time objects to datetime objects (add today's date)
	datetimes = np.array([datetime.combine(date.today(), t) for t in timestamps])
	time_hours = mdates.date2num(datetimes)
	
	# Clip spectra to maximum power level
	spectra_clipped = np.clip(spectra, -80, -20)

	# Format y-axis as time
	plt.gca().yaxis_date()
	date_form = mdates.DateFormatter('%H:%M:%S')
	plt.gca().yaxis.set_major_formatter(date_form)
	plt.gca().yaxis.set_major_locator(mdates.HourLocator(interval=1))

	# Create heatmap
	plt.imshow(spectra_clipped, aspect='auto', origin='lower', 
				extent=[faxis_mhz[0], faxis_mhz[-1], time_hours[0], time_hours[-1]],
				cmap='viridis', interpolation='nearest')
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('Time (hours)') 

	# Set y-ticks to show only full hour and half-hour
	# y_ticks = np.arange(np.floor(time_hours[0]), np.ceil(time_hours[-1]), 0.5)
	# plt.yticks(y_ticks)
	plt.title(title)
	cbar = plt.colorbar(label='Power (dBm)')

	# Set color limits for the heatmap
	plt.clim(vmin=-80, vmax=-20)
	# plt.ylim(y_ticks[0], y_ticks[-1])

	if output_path:
		plt.savefig(output_path, dpi=150, bbox_inches='tight')
		logging.info("Heatmap saved to %s", output_path)
	
	if show_plot:
		plt.show()
	else:
		plt.close()

def __main__(date_dir, state_indx=0):
    loaded = get_date_state_specs(date_dir, state_indx=state_indx)
    timestamps, spectra = read_loaded(loaded, sort='ascending')
    plot_waterfall_heatmap(timestamps, spectra, faxis, 
						   title=f'Waterfall Plot {os.path.basename(date_dir)}: State {state_indx}',
						   output_path=pjoin(date_dir, f'waterfall_state{state_indx}.png'),
						   show_plot=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python waterfall_plotter.py <date_directory> [state_index]")
        sys.exit(1)
  
    input_dir = sys.argv[1]
    state_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    __main__(input_dir, state_indx=state_index)
    
    


	
	


  
  
  

	 
 
