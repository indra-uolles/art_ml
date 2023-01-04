# -*- coding: utf-8 -*-
import numpy as np

def circular_median(signal):
    # angles should be in radians
    median_cos = np.median(np.cos(signal))
    median_sin = np.median(np.sin(signal))
    x = np.arctan2(median_sin, median_cos)
    return x