 # -*- coding: utf-8 -*-

from colormath.color_objects import sRGBColor, LCHuvColor
from colormath.color_conversions import convert_color
import numpy as np
from PIL import Image
from functools import cmp_to_key

def get_rgb_colors_arr(steps):
    rgb_colors = []
    for i in range(len(steps)):
        for j in range(len(steps)):
            for k in range(len(steps)):
                rgb_colors.append([steps[i], steps[j], steps[k]])
    return rgb_colors

def rgb2lch(r, g, b):
    lch_color = convert_color(sRGBColor(r/255, g/255, b/255), LCHuvColor)
    return lch_color

def lch2hue(lch_color):
    h = lch_color.lch_h

    if h <= 24:
        return 'VR'
    elif h > 24 and h <= 38:
        return 'R'
    elif h > 38 and h <= 53:
        return 'RO'
    elif h > 53 and h <= 65:
        return 'O'
    elif h > 65 and h <= 80:
        return 'YO'
    elif h > 80 and h <= 90:
        return 'OY'
    elif h > 90 and h <= 100:
        return 'Y'
    elif h > 100 and h <= 115:
        return 'GY'
    elif h > 115 and h <= 130:
        return 'YG'
    elif h > 130 and h <= 145:
        return 'YG'
    elif h > 145 and h <= 162:
        return 'G'
    elif h > 162 and h <= 180:
        return 'G'
    elif h > 180 and h <= 204:
        return 'BG'
    elif h > 204 and h <= 218:
        return 'BG'
    elif h > 218 and h <= 233:
        return 'GB'
    elif h > 233 and h <= 245:
        return 'GB'
    elif h > 245 and h <= 260:
        return 'B'
    elif h > 260 and h <= 270:
        return 'B'
    elif h > 270 and h <= 280:
        return 'VB'
    elif h > 280 and h <= 295:
        return 'VB'
    elif h > 295 and h <= 310:
        return 'BV'
    elif h > 310 and h <= 325:
        return 'V'
    elif h > 325 and h <= 342:
        return 'RV'
    elif h > 342:
        return 'VR'

def belongs_to_hue(lch_color, hue):
    if lch2hue(lch_color) == hue:
        return True
    return False

def filter_rgb_colors_by_hue(rgb_colors, hue):
    filtered_rgb_colors = []
    for rgb_color in rgb_colors:
        lch_color = rgb2lch(rgb_color[0], rgb_color[1], rgb_color[2])
        if (belongs_to_hue(lch_color, hue)):
            filtered_rgb_colors.append(rgb_color)
    return filtered_rgb_colors

def filter_rgb_colors_by_lightness(rgb_colors, l, delta):
    filtered_rgb_colors = []
    for rgb_color in rgb_colors:
        lch_color = rgb2lch(rgb_color[0], rgb_color[1], rgb_color[2])
        if (abs(lch_color.lch_l - l) <= delta):
            filtered_rgb_colors.append(rgb_color)
    return filtered_rgb_colors

def sort_rgb_colors_by_chroma(rgb_colors):
    def compare(i1, i2):
        lch_color1 = rgb2lch(i1[0], i1[1], i1[2])
        lch_color2 = rgb2lch(i2[0], i2[1], i2[2])
        return lch_color2.lch_c - lch_color1.lch_c
    rgb_colors.sort(key=cmp_to_key(compare))
    return rgb_colors

def generate_rgb_image_arr(width, height, rgb_color):
    rgb_image = np.zeros((width, height, 3), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            rgb_image[x, y, 0] = rgb_color.rgb_r
            rgb_image[x, y, 1] = rgb_color.rgb_g
            rgb_image[x, y, 2] = rgb_color.rgb_b
    return rgb_image

def show_image(rgb_color, width, height):
    image_arr_rgb = generate_rgb_image_arr(width, height, rgb_color)
    image = Image.fromarray(image_arr_rgb).convert('RGBA')
    image.show()