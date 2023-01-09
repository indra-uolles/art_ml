# -*- coding: utf-8 -*-
# python images.test.py

import unittest
import numpy as np
import math
import hist
from colormath.color_objects import sRGBColor, LCHabColor
# import json

class TestHist(unittest.TestCase):

    def test_circular_median(self):
        signal = [math.radians(math.pi/8), math.radians(math.pi/4), math.radians(3*math.pi/8)]
        self.assertEqual(hist.circular_median(signal), math.radians(math.pi/4))  

    def test_hue_hist(self):
        colors = [
            LCHabColor(11, 32, 359),
            LCHabColor(14, 36, 356),
            LCHabColor(18, 41, 353),
            LCHabColor(16, 100, 139),
        ] 
        # print(json.dumps(hist.hue_hist(colors), indent = 4))
        # one saturated color almost beats the other 3 not very saturated colors
        self.assertEqual(5, 5)
        np.testing.assert_equal(hist.hue_hist(colors), {
            "VR": 27.225,
            "R": 0,
            "RO": 0,
            "O": 0,
            "YO": 0,
            "OY": 0,
            "Y": 0,
            "GY": 0,
            "YG": 0,
            "YG2": 25.0,
            "G": 0,
            "G2": 0,
            "BG": 0,
            "BG2": 0,
            "GB": 0,
            "GB2": 0,
            "B": 0,
            "B2": 0,
            "VB": 0,
            "VB2": 0,
            "BV": 0,
            "V": 0,
            "RV": 0
        }) 

    def test_draw_hue_hist(self):
        data = {
            "VR": [LCHabColor(14, 36, 356), 27.225],
            "YG2": [LCHabColor(16, 100, 139), 10.0],
        }
        hist.draw_hue_hist(data).show()
        self.assertEqual(5, 5)  

if __name__ == "__main__":
    unittest.main()