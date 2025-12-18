import os
import matplotlib.pyplot as plt

LEGEND = ['Antenna', 'Open Circuit', 'Short', 'Long cable short',
          'Black body', 'Ambient temperature load', 'Noise diode', 'Long cable open']
LEGEND_WO_ANTENNA = ['Open Circuit', 'Short', 'Long cable short', 'Black body', 'Ambient temperature load', 'Noise diode', 'Long cable open']

DATA_PATH = '/media/peterson/INDURANCE'

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
COLOR_CODE = {'Antenna': color_cycle[0], 'Open Circuit': color_cycle[1], 'Short': color_cycle[2],
              'Long cable short': color_cycle[3], 'Black body': color_cycle[4],
              'Ambient temperature load': color_cycle[5], 'Noise diode': color_cycle[6],
              'Long cable open': color_cycle[7]}

def map_filename_to_legend(statename):
    """Map the spectrum state name from filename to legend name."""
    mapping = {'state0': 'Antenna', 'state1': 'Open Circuit',
               'state2': 'Short', 'state3': "Long cable short",
               'state4': 'Black body', 'state5': 'Ambient temperature load',
               'state6': 'Noise diode', 'state7': "Long cable open",
               'stateOC': '6" open'}
    return mapping.get(statename, statename)

def parse_filename(spec_path) -> tuple[str, str, str]:
    """Parse the spectrum file name to extract state_name, antenna_name, and time_stamp."""
    pbase = os.path.basename
    filename = pbase(spec_path).split('.')[0]
    spec_state = filename.split('_')[-1]
    antenna_name = filename.split('_')[-2]
    time_stamp = filename.split('_')[-3]

    return spec_state, antenna_name, time_stamp