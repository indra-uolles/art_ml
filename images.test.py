# -*- coding: utf-8 -*-
# python images.test.py

import unittest
import images

class TestImages(unittest.TestCase):
    def test_intensity():
        intensive_warm_green = [68, 220, 101]
        less_intensive_warm_green = [63, 186, 106]
        intensive_yellow = [225, 201, 40]
        less_intensive_yellow = [204, 178, 18]
        intensive_orange = [250, 122, 5]
        less_intensive_orange = [198, 125, 0]

        assert images.get_color_intensity(intensive_warm_green) == 160.986, 'should be 160.986'
        assert images.get_color_intensity(intensive_yellow) == 189.822, 'should be 189.822'
        assert images.get_color_intensity(intensive_orange) == 146.93399999999997, 'should be 146.93399999999997'

        assert images.get_color_intensity(less_intensive_warm_green) == 140.10299999999998, 'should be 140.10299999999998'
        assert images.get_color_intensity(less_intensive_yellow) == 167.53399999999996, 'should be 167.53399999999996'
        assert images.get_color_intensity(less_intensive_orange) == 132.577, 'should be 132.577'

        assert images.get_color_intensity_2(intensive_warm_green) == 129.66666666666666, 'should be 129.66666666666666'
        assert images.get_color_intensity_2(intensive_yellow) == 155.33333333333334, 'should be 155.33333333333334'
        assert images.get_color_intensity_2(intensive_orange) == 125.66666666666667, 'should be 125.66666666666667'

        assert images.get_color_intensity_2(less_intensive_warm_green) == 118.33333333333333, 'should be 118.33333333333333'
        assert images.get_color_intensity_2(less_intensive_yellow) == 133.33333333333334, 'should be 133.33333333333334'
        assert images.get_color_intensity_2(less_intensive_orange) == 107.66666666666667, 'should be 107.66666666666667'


if __name__ == "__main__":
    TestImages.test_intensity()