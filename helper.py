from matplotlib.widgets import Slider
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import os
import time
import threading
import heapq


def dbm_convert(spectrum):
    return [10 * np.log10(s) - 135 for s in spectrum]

