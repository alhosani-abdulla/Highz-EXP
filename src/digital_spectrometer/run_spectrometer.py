# =======================
# Imports
# =======================
import os
import time
import struct
import argparse
from datetime import datetime

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import casperfpga
import rcal

# =======================
# CONFIGURATION
# =======================

# Paths and hardware configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '../../'))
# Update the subfolder and filename as needed:
CONFIG_PATH = os.path.join(repo_root, 'fpga_config', '03-11-2025', 'v26.fpg')
BASE_PATH = '/media/peterson'  # Path to external storage for data

# Acquisition and instrument parameters
FPGA_IP = '169.254.223.21'   # FPGA IP address (link local address when connected via ethernet)
ANTENNA = '1'                # Antenna identifier

# FFT parameters
ACC_LENGTH = 8750            # Accumulation length (see notes below)
NCHANNELS = 4                # Number of channels
NFFT = 32768                 # FFT length

# Calibration and observation parameters
CAL_ACC_N = 15                # Number of spectra per calibration state per cycle
ANT_ACC_N = 15                # Number of spectra for antenna state per cycle
FB_N = 0                     # Number of spectra for filter bank calibration
SAVE_EACH_ACC = False        # True: save each accumulation, False: sum accumulations

# The time in seconds for one accumulation. Skips one accumulation when switching states.
SWITCH_DELAY = ACC_LENGTH * 3 / 100000 

# =======================
# End CONFIGURATION
# =======================

########################################################################


def initialize_fpga():
  """
  Initialize the FPGA and ADC, program the bitstream, and set up clocks.
  Returns the fpga and adc objects.
  """
  print(datetime.utcfromtimestamp(time.time()))
  rcal.gpio_switch(0, 2)
  try:
    print('Running Script ...')
    # Connect to the FPGA
    fpga = casperfpga.CasperFpga(FPGA_IP)
    fpga.upload_to_ram_and_program(CONFIG_PATH)

    adc = fpga.adcs['rfdc']
    adc.init()
    print(adc.status())

    c = adc.show_clk_files()
    adc.progpll('lmk', c[1])
    adc.progpll('lmx', c[0])

  except Exception as e:
    print(f'An error occurred during FPGA initialization: {e}')
    time.sleep(180)
    os.system('sudo reboot')

  time.sleep(15)  # Let clocks adjust

  # Set accumulation length
  fpga.write_int('acc_len', ACC_LENGTH)
  print('Initialization complete. Taking data...')
  return fpga, adc

def main():
  global acc_n, cycle_count, sub_dir_count, current_parent_dir, current_sub_dir_path, run_directory

  # Initialize global variables
  cycle_count = 0
  sub_dir_count = 0
  current_parent_dir = None # name of the current parent directory
  current_sub_dir_path = None  # Path to the current subdirectory

  # Initialize FPGA and ADC
  fpga, adc = initialize_fpga()

  # Initial GPIO/Switch state
  rcal.gpio_switch(1,SWITCH_DELAY)
  
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='high-z digital spectrometer')
  parser.add_argument('--state', default=None, type=int, help='switch state (0-10)')
  parser.add_argument('--run_directory', default=None, help='parent directory name for this observing run')
  args = parser.parse_args()
  print(args, flush=True)
  input_state = args.state
  run_directory = args.run_directory

  # Main data acquisition loop
  while True:
    # Check if storage device is attached!
    while next(os.walk(BASE_PATH))[1] == []:
      print('No drive attached. Please attach drive to continue taking data.')
      time.sleep(5) #check every 5 seconds for an attached drive
    
    acc_n = fpga.read_uint('acc_cnt')
    
    if input_state is not None:
      rcal.gpio_switch(input_state, SWITCH_DELAY)
      for n in range(100): #collect 100 spectra in the specified state
        data, name = save_all_data(fpga, switch_value=input_state)
      print(f'Completed data collection for state {input_state}. Exiting.')
      break
    else:
      #Digital Spectrometer Calibaration Sweep.
      for s in range(2,11):
        spectra_n = CAL_ACC_N
        if s == 8: # switching to open circuit
          rcal.gpio_switch(1, SWITCH_DELAY)
          s = 'OC'
        elif s == 9: # switching to state 0 (Antenna - Main Switch Powered)
          s = 0
          rcal.gpio_switch(s, SWITCH_DELAY)
        elif s == 10: # Filter Bank Calibration State
          rcal.gpio_switch(s, SWITCH_DELAY)
          s = 'FB'
          spectra_n = FB_N
        else:
          rcal.gpio_switch(s, SWITCH_DELAY) # Switching Calibration States
        
        for n in range(spectra_n):
          data, name = save_all_data(fpga, switch_value=s)

      #Observing with the Antenna - collecting ant_acc_n spectras 
      state = 1
      rcal.gpio_switch(state, SWITCH_DELAY)
      for cnt in range(ANT_ACC_N):
        data, name = save_all_data(fpga, switch_value=state)

    # Increment the cycle count after one full calibration + observation cycle
    cycle_count += 1

def print_acc_cnt(fpga, last_acc_n):
  acc_n = fpga.read_uint('acc_cnt')
  if acc_n > last_acc_n: 
    last_acc_n = acc_n

  c = 1
  while acc_n == last_acc_n:
    acc_n = fpga.read_uint('acc_cnt')
    c += 1
  
  #c = 0
  #start_time = time.time()
  while False:
    current_cnt = fpga.read_uint('acc_cnt')
    current_time = time.time()
    c += 1
    if current_cnt > acc_n:
      increment_time = current_time - start_time
      print(f'Time for increment: {increment_time} seconds {c}')
    
      start_time = current_time
      acc_n = current_cnt
      c = 0
  print(acc_n, c)
  return acc_n
  
def get_vacc_data(fpga):
  half_nfft = NFFT//2
  chunk = half_nfft//NCHANNELS
  raw = np.zeros((NCHANNELS, chunk))
  
  start_time = time.time()
  for i in range(NCHANNELS):
    q = fpga.read('q{:d}'.format(i+1), chunk * 8, 0)
    raw[i,:] = struct.unpack('>{:d}Q'.format(chunk), q)

  interleave_q = []
  for i in range(chunk):
    for j in range(NCHANNELS):
      interleave_q.append(raw[j,i])
  print(f'data read time: {time.time() - start_time}')

  return interleave_q, fpga.read_uint('acc_cnt')

def write_filename(state, acc_n):
  basename = str(datetime.utcfromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S') + f'_antenna{ANTENNA}_state{state}')
  
  return f"{basename}_{acc_n}" if SAVE_EACH_ACC else basename

def sum_spectrum(spectrum):
  return np.array([sum(x) for x in zip(*spectrum)], dtype=np.float64)

def save_data(dataDict, filename):
  """
  Save data into the appropriate directory.
  """
  global current_parent_dir, sub_dir_count, current_sub_dir_path, cycle_count, run_directory
  
  dirnames = next(os.walk(BASE_PATH))[1]
  while not dirnames:
    print('No drive attached. Please attach a drive to continue taking data.')
    time.sleep(5)
    dirnames = next(os.walk(BASE_PATH))[1]
  dirname = dirnames[-1]

  # Determine the parent directory for this run
  if run_directory is not None:
    # Use run_directory as the parent directory for this run
    parent_dir_name = run_directory
  else:
    parent_dir_name = filename.split('_')[0]

  parent_dir_path = os.path.join(BASE_PATH, dirname, parent_dir_name)
  # Create the parent directory if it doesn't exist
  if parent_dir_name not in os.listdir(os.path.join(BASE_PATH, dirname)):
    os.mkdir(parent_dir_path)

  # Create a new subdirectory only after completing a cycle
  if cycle_count >= 1 or current_sub_dir_path is None:
    current_parent_dir = parent_dir_name
    current_sub_dir_path, sub_dir_count = get_sub_directory(parent_dir_path, sub_dir_count)
    cycle_count = 0  # Reset cycle count after creating a new subdirectory

  # Save the file in the appropriate directory
  file_path = os.path.join(current_sub_dir_path, f"{filename}.npy")
  np.save(file_path, dataDict, allow_pickle='False')
  print(f"Data saved to {file_path}")

def save_all_data(fpga, switch_value):
  global acc_n
  t_1 = time.time()
  #save volt + temp data
  #dataDict = volt_therm_lib.get_temp_volt_data()
  dataDict = {}
  dataDict["switch state"] = switch_value
  t_2 = time.time()
  if SAVE_EACH_ACC:
    # Save each individual accumulation
    acc_n = print_acc_cnt(fpga, acc_n)
    t_3 = time.time()
    s, cnt = get_vacc_data(fpga)
    t_4 = time.time()
    dataDict["spectrum"] = s
    filename = write_filename(switch_value, cnt)
    t_5 = time.time()
    save_data(dataDict, filename)
    
  else:
    spectra = {}
    while len(spectra) < 3:
      acc_n = print_acc_cnt(fpga, acc_n)
      s, cnt = get_vacc_data(fpga)
      spectra[cnt] = s
    
    spectra_list = [spectra[key] for key in spectra.keys()]
    
    #separating corrupt spectra from clean ones:
    #clean_spectra, corrupt_spectra = detect_periodic_noise(spectra_list)
    
    dataDict["spectrum"] = sum_spectrum(spectra_list)
    #dataDict["nspectra"] = len(clean_spectra)
    filename = write_filename(switch_value, 'average',)
    save_data(dataDict, filename)

    print(dataDict.keys())
    print(switch_value)
    print(dataDict["switch state"])
    #print("Number of Clean Spectra:" + dataDict["nspectra"])
  t_6 = time.time()

  #print(f'save_all_data: {t_2-t_1}, {t_3-t_2}, {t_4-t_3}, {t_5-t_4}, {t_6-t_5}')
  print(f'save_all_data: {t_2-t_1}, {t_6-t_2}')

  return dataDict, filename

def get_sub_directory(parent_dir_path, sub_dir_count):
    """
    Handles directory creation and naming for observing runs.

    Parameters:
        base_path (str): Parent directory path where subdirectories are stored.
        sub_dir_count (int): Counter for subdirectory numbers.

    Returns:
        str: Path to the subdirectory for the current data chunk.
    """

    # Determine the unique subdirectory name
    sub_dir_name = f"{datetime.utcfromtimestamp(time.time()).strftime('%H%M%S')}"
    sub_dir_path = os.path.join(parent_dir_path, sub_dir_name)

    # Create subdirectory if it doesn't exist
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
        sub_dir_count += 1

    return sub_dir_path, sub_dir_count

########################################################################

if __name__ == "__main__":
  main()
