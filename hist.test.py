# -*- coding: utf-8 -*-
# python images.test.py

import unittest
import numpy as np
import math
import hist

class TestHist(unittest.TestCase):

    def test_circular_median(self):
        signal = [math.radians(math.pi/8), math.radians(math.pi/4), math.radians(3*math.pi/8)]
        self.assertEqual(hist.circular_median(signal), math.radians(math.pi/4))

if __name__ == "__main__":
    unittest.main()