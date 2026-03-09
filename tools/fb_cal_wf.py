from highz_exp.sys_cal import SystemCalibrationProcessor
from matplotlib.pylab import Any

class FBCalibrationProcessor(SystemCalibrationProcessor):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
    