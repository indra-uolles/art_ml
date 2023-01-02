# -*- coding: utf-8 -*-
# python images.test.py

import unittest
import images as images
import numpy as np
from colormath.color_objects import LCHabColor, sRGBColor

class TestImages(unittest.TestCase):

    def test_get_rgb_colors_arr(self):
        rgb_colors = images.get_rgb_colors_arr([0, 20])
        np.testing.assert_array_equal(rgb_colors, [[0, 0, 0], [0, 0, 20], [0, 20, 0], [0, 20, 20], [20, 0, 0], [20, 0, 20], [20, 20, 0], [20, 20, 20]])

    def test_rgb2lch(self):
        np.testing.assert_allclose(images.rgb2lch(0, 196, 92).get_value_tuple(), [69.647286,  75.498757, 147.558313])

    def test_lch2hue(self):
        self.assertEqual(images.lch2hue(LCHabColor(69.647286,  75.498757, 147.558313)), 'G')

    def test_belongs_to_hue(self):
        self.assertTrue(images.belongs_to_hue(LCHabColor(69.647286,  75.498757, 147.558313), 'G'))

    def test_filter_rgb_colors_by_hue(self):
        rgb_colors = [
            [255, 0, 0], [255, 128, 0], [255, 255, 0], [128, 255, 0], [0, 255, 0], [0, 255, 128],
            [0, 255, 255], [0, 128, 255], [0, 0, 255], [127, 0, 255], [255, 0, 255], [255, 0, 127]
        ]
        filtered_rgb_colors = images.filter_rgb_colors_by_hue(rgb_colors, 'G')
        np.testing.assert_allclose(filtered_rgb_colors, [[0, 255, 128]])

    def test_generate_rgb_image_arr(self):
        # images.show_image(sRGBColor(0, 196, 92), 100, 100)
        image = images.generate_rgb_image_arr(1, 1, sRGBColor(128, 255, 0))
        np.testing.assert_array_equal(image, [[[128, 255, 0]]])

    def test_filter_rgb_colors_by_lightness(self):
       rgb_colors = [[255, 0, 0], [255, 128, 0], [255, 255, 0], [128, 255, 0], [0, 196, 92]]
       filtered_rgb_colors = images.filter_rgb_colors_by_lightness(rgb_colors, 69.6, 0.05)
       np.testing.assert_array_equal(filtered_rgb_colors, [[0, 196, 92]])

    def test_sort_rgb_colors_by_chroma(self):
        rgb_colors = [[126, 239, 160], [68, 240, 120], [115, 239, 153], [93, 240, 138]]
        sorted_rgb_colors = images.sort_rgb_colors_by_chroma(rgb_colors)
        np.testing.assert_array_equal(sorted_rgb_colors, [[68, 240, 120], [93, 240, 138], [115, 239, 153], [126, 239, 160]])

    def test_get_label_for_image(self):
        self.assertEqual(images.get_label_for_image(sRGBColor(0, 196, 92)), '0,196,92\n70,75,148')
        # compared lch result with with https://css.land/lch/
        # images.get_rgb_image_with_label(100, 100, sRGBColor(0, 196, 92)).show()

    def test_get_unique_l_values(self):
        rgb_colors = [[126, 239, 160], [68, 240, 120], [115, 239, 153], [93, 240, 138]]
        self.assertEqual(images.get_unique_l_values(rgb_colors), [86, 85, 84])

    def test_get_extended_list(self):
        colors = [[245, 138, 1]]
        extended_colors = images.get_extended_list(colors, [0, 0, 0], 3)
        np.testing.assert_array_equal(extended_colors, [[245, 138, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_get_mid_hue(self):
        self.assertEqual(images.get_mid_hue(38, 53), 8)

    def test_is_out_of_RGB_gamut(self):
        self.assertTrue(images.is_out_of_RGB_gamut(sRGBColor(0, 1.1, 0)))
        self.assertFalse(images.is_out_of_RGB_gamut(sRGBColor(0, 1, 0)))

    def test_get_rgb_colors_arr_with_hue(self):
        rgb_colors = images.get_rgb_colors_arr_with_hue(0, 20, 'G')
        np.testing.assert_array_equal(rgb_colors, [[0, 255, 128]])

    def test_lch2rgb(self):
        # close enough to [0, 196, 92]
        np.testing.assert_array_equal(images.lch2rgb(69.647286,  75.498757, 147.558313).get_upscaled_value_tuple(), [0, 197, 90])

        images.show_ch_swatch(50)

if __name__ == "__main__":
    unittest.main()