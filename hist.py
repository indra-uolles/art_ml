# -*- coding: utf-8 -*-
import numpy as np
import lch
from PIL import Image, ImageFont, ImageDraw, ImageOps
from colormath.color_objects import sRGBColor, LCHabColor, LabColor

def circular_median(signal):
    # angles should be in radians
    median_cos = np.median(np.cos(signal))
    median_sin = np.median(np.sin(signal))
    x = np.arctan2(median_sin, median_cos)
    return x

def hue_hist(lch_colors):
    hues = ['VR', 'R', 'RO', 'O', 'YO', 'OY','Y', 'GY', 'YG', 'YG2', 'G', 'G2', 'BG', 'BG2', 'GB',
    'GB2', 'B', 'B2', 'VB', 'VB2', 'BV', 'V', 'RV', 'VR']
    hist = {}
    size = len(lch_colors)
    for hue in hues:
        filtered_colors = lch.filter_colors_by_hue(lch_colors, hue)
        if (len(filtered_colors) == 0):
            hist[hue] = 0
        else:
            chromas = np.array([color.lch_c for color in filtered_colors])
            rad_angles = np.array([np.deg2rad(color.lch_h) for color in filtered_colors])
            cos = np.cos(rad_angles)
            sin = np.sin(rad_angles)
            a_s = np.dot(cos, chromas)
            b_s = np.dot(sin, chromas)
            r_n = np.sqrt(a_s**2 + b_s**2)
            hist[hue] = round(r_n, 1)/size

    return hist
    
def draw_hue_hist(data):
    sum = 0
    for hue in data:
        sum += data[hue][1]

    for hue in data:
        data[hue][1] = round(data[hue][1]*100/sum, 0)

    # scale*min >= 10px  
    min = np.min([data[hue][1] for hue in data])
    if (min < 10):
        scale = 10/min
    else: 
        scale = 1    

    image_height = 0
    for hue in data:
        image_height += data[hue][1]*scale 

    image = Image.new('RGBA', (100, int(image_height)), (0, 0, 0))
    offset = 0
    for hue in data:
        lch_color = data[hue][0]
        l, c, h = lch_color.lch_l, lch_color.lch_c, lch_color.lch_h
        rgb_color = lch.lch2rgb(l, c, h)
        upscaled_rgb_color = sRGBColor(rgb_color.rgb_r*255, rgb_color.rgb_g*255, rgb_color.rgb_b*255)
        _image = lch.get_rgb_image(100, int(scale*data[hue][1]*100), upscaled_rgb_color)
        image.paste(_image, (0, offset))
        offset += int(image_height - data[hue][1]*scale)

    return image  