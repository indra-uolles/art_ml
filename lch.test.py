# -*- coding: utf-8 -*-
# python images.test.py

import unittest
import lch as lch
import numpy as np
from colormath.color_objects import LCHabColor, sRGBColor

class TestImages(unittest.TestCase):

    def test_get_hue_ranges(self):
        self.assertEqual(lch.get_hue_ranges('VR'), [340, 360])

    def test_filter_colors_by_hue(self):
        # 3 reds and one green
        def value(n):
            return n.get_value_tuple()    
        colors = [
            LCHabColor(11, 32, 359),
            LCHabColor(14, 36, 356),
            LCHabColor(18, 41, 353),
            LCHabColor(12, 14, 247),
        ]
        colors_values = list(map(value, lch.filter_colors_by_hue(colors, 'VR')))
        np.testing.assert_allclose(colors_values, [LCHabColor(11, 32, 359).get_value_tuple(), LCHabColor(14, 36, 356).get_value_tuple(), LCHabColor(18, 41, 353).get_value_tuple()])    

    def test_get_rgb_colors_arr(self):
        rgb_colors = lch.get_rgb_colors_arr([0, 20])
        np.testing.assert_array_equal(rgb_colors, [[0, 0, 0], [0, 0, 20], [0, 20, 0], [0, 20, 20], [20, 0, 0], [20, 0, 20], [20, 20, 0], [20, 20, 20]])

    def test_rgb2lch(self):
        np.testing.assert_allclose(lch.rgb2lch(0, 196, 92).get_value_tuple(), [69.647286,  75.498757, 147.558313])

    def test_lch2hue(self):
        self.assertEqual(lch.lch2hue(LCHabColor(69.647286,  75.498757, 147.558313)), 'G')

    def test_belongs_to_hue(self):
        self.assertTrue(lch.belongs_to_hue(LCHabColor(69.647286,  75.498757, 147.558313), 'G'))

    def test_generate_rgb_image_arr(self):
        # images.show_image(sRGBColor(0, 196, 92), 100, 100)
        image = lch.generate_rgb_image_arr(1, 1, sRGBColor(128, 255, 0))
        np.testing.assert_array_equal(image, [[[128, 255, 0]]])

    def test_get_label_for_image(self):
        self.assertEqual(lch.get_label_for_image(sRGBColor(0, 196, 92)), '0,196,92\n70,75,148')
        # compared lch result with with https://css.land/lch/
        # images.get_rgb_image_with_label(100, 100, sRGBColor(0, 196, 92)).show()

    def test_get_unique_l_values(self):
        rgb_colors = [[126, 239, 160], [68, 240, 120], [115, 239, 153], [93, 240, 138]]
        self.assertEqual(lch.get_unique_l_values(rgb_colors), [86, 85, 84])

    def test_is_out_of_RGB_gamut(self):
        self.assertTrue(lch.is_out_of_RGB_gamut(sRGBColor(0, 1.1, 0)))
        self.assertFalse(lch.is_out_of_RGB_gamut(sRGBColor(0, 1, 0)))

    def test_lch2rgb(self):
        # close enough to [0, 196, 92]
        np.testing.assert_array_equal(lch.lch2rgb(69.647286,  75.498757, 147.558313).get_upscaled_value_tuple(), [0, 197, 90])

        lch.show_ch_swatch(40)

if __name__ == "__main__":
    unittest.main()