# -*- coding: utf-8 -*-
# python color_sys.test.py

import unittest
import color_sys as color_sys
import numpy as np
from colormath.color_objects import LCHabColor, sRGBColor

class TestImages(unittest.TestCase):

    def test_get_hue_ranges(self):
        self.assertEqual(color_sys.get_hue_ranges('VR'), [340, 360])

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
        colors_values = list(map(value, color_sys.filter_colors_by_hue(colors, 'VR')))
        np.testing.assert_allclose(colors_values, [LCHabColor(11, 32, 359).get_value_tuple(), LCHabColor(14, 36, 356).get_value_tuple(), LCHabColor(18, 41, 353).get_value_tuple()])    

    def test_get_rgb_colors_arr(self):
        rgb_colors = color_sys.get_rgb_colors_arr([0, 20])
        np.testing.assert_array_equal(rgb_colors, [[0, 0, 0], [0, 0, 20], [0, 20, 0], [0, 20, 20], [20, 0, 0], [20, 0, 20], [20, 20, 0], [20, 20, 20]])

    def test_rgb2lch(self):
        np.testing.assert_allclose(color_sys.rgb2lch(0, 196, 92).get_value_tuple(), [69.647286,  75.498757, 147.558313])

    def test_lch2hue(self):
        self.assertEqual(color_sys.lch2hue(LCHabColor(69.647286,  75.498757, 147.558313)), 'G')

    def test_belongs_to_hue(self):
        self.assertTrue(color_sys.belongs_to_hue(LCHabColor(69.647286,  75.498757, 147.558313), 'G'))

    def test_generate_rgb_image_arr(self):
        # images.show_image(sRGBColor(0, 196, 92), 100, 100)
        image = color_sys.generate_rgb_image_arr(1, 1, sRGBColor(128, 255, 0))
        np.testing.assert_array_equal(image, [[[128, 255, 0]]])

    def test_get_label_for_image(self):
        self.assertEqual(color_sys.get_label_for_image(sRGBColor(0, 196, 92)), '0,196,92\n70,75,148')
        # compared lch result with with https://css.land/lch/
        # images.get_rgb_image_with_label(100, 100, sRGBColor(0, 196, 92)).show()

    def test_get_unique_l_values(self):
        rgb_colors = [[126, 239, 160], [68, 240, 120], [115, 239, 153], [93, 240, 138]]
        self.assertEqual(color_sys.get_unique_l_values(rgb_colors), [86, 85, 84])

    def test_is_out_of_RGB_gamut(self):
        self.assertTrue(color_sys.is_out_of_RGB_gamut(sRGBColor(0, 1.1, 0)))
        self.assertFalse(color_sys.is_out_of_RGB_gamut(sRGBColor(0, 1, 0)))

    def test_lch2rgb(self):
        # close enough to [0, 196, 92]
        np.testing.assert_array_equal(color_sys.lch2rgb(69.647286,  75.498757, 147.558313).get_upscaled_value_tuple(), [0, 197, 90])

        color_sys.show_ch_swatch(40)

    def test_std_deviation_rgb(self):
        # dark blue, looks like black
        self.assertEqual(color_sys.std_deviation_rgb(sRGBColor(10,10,40)), 0.0554593553871802)
        # lch_color_dark_blue = color_sys.rgb2lch(10, 10, 40).get_value_tuple()
        # print('lch_color dark blue', str(lch_color_dark_blue))
        # (3.9273987556853136, 20.677351101688785, 293.7620180683244) lightness is low, < 50

        # light gray
        self.assertEqual(color_sys.std_deviation_rgb(sRGBColor(200,198,200)), 0.0036972903591453335)
        lch_color_light_gray = color_sys.rgb2lch(200, 198, 200).get_value_tuple()
        print('lch_color light gray', str(lch_color_light_gray))
        # (80.08821501569784, 1.3019778933573856, 324.2314996385326) lightness is high, but chroma is low, < 50

        # eye hurting green
        self.assertEqual(color_sys.std_deviation_rgb(sRGBColor(23,212,74)), 0.31308758676411097)
        lch_color_eye_hurting_green = color_sys.rgb2lch(23,212,74).get_value_tuple()
        print('lch_color hurting green', str(lch_color_eye_hurting_green))
        # (74.67375822259342, 88.64935960491266, 142.06265494221583) lightness is high, chroma is high,
        # it's eye hurting

    def test_chromatic_membership_degree(self):
        self.assertEqual(color_sys.chromatic_membership_degree(sRGBColor(10, 10, 40)), 0)
        self.assertEqual(color_sys.chromatic_membership_degree(sRGBColor(200, 198, 200)), 0)
        self.assertEqual(color_sys.chromatic_membership_degree(sRGBColor(23,212,74)), 1)

if __name__ == "__main__":
    unittest.main()