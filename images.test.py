# -*- coding: utf-8 -*-
# python images.test.py

import unittest
import images as images
import numpy as np
from colormath.color_objects import LCHuvColor

class TestImages(unittest.TestCase):
    def test_get_rgb_colors_arr(self):
        rgb_colors = images.get_rgb_colors_arr([0, 20])
        np.testing.assert_array_equal(rgb_colors, [[0, 0, 0], [0, 0, 20], [0, 20, 0], [0, 20, 20], [20, 0, 0], [20, 0, 20], [20, 20, 0], [20, 20, 20]])

    def test_rgb2lch(self):
        np.testing.assert_allclose(images.rgb2lch(0, 196, 92).get_value_tuple(), [69.647286,  87.940279, 135.733355])

    def test_lch2hue(self):
        self.assertEqual(images.lch2hue(LCHuvColor(69.647286,  87.940279, 135.733355)), 'YG')

    def test_belongs_to_hue(self):
        self.assertTrue(images.belongs_to_hue(LCHuvColor(69.647286,  87.940279, 135.733355), 'YG'))

    def test_filter_rgb_colors_by_hue(self):
        rgb_colors = [
            [255, 0, 0], [255, 128, 0], [255, 255, 0], [128, 255, 0], [0, 255, 0], [0, 255, 128],
            [0, 255, 255], [0, 128, 255], [0, 0, 255], [127, 0, 255], [255, 0, 255], [255, 0, 127]
        ]
        filtered_rgb_colors = images.filter_rgb_colors_by_hue(rgb_colors, 'YG')
        np.testing.assert_allclose(filtered_rgb_colors, [[128, 255, 0], [0, 255, 0], [0, 255, 128]])

if __name__ == "__main__":
    unittest.main()