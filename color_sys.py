 # -*- coding: utf-8 -*-

from colormath.color_objects import sRGBColor, LCHabColor, LabColor
from colormath.color_conversions import convert_color
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
from functools import cmp_to_key
import matplotlib.pyplot as plt

def get_rgb_colors_arr(steps):
    rgb_colors = []
    for i in range(len(steps)):
        for j in range(len(steps)):
            for k in range(len(steps)):
                rgb_colors.append([steps[i], steps[j], steps[k]])
    return rgb_colors

def rgb2lch(r, g, b):
    lab_color = convert_color(sRGBColor(r/255, g/255, b/255), LabColor)
    lch_color = convert_color(lab_color, LCHabColor)
    return lch_color 

def lch2rgb(l, c, h):
    lch_color = LCHabColor(l, c, h)
    lab_color = convert_color(lch_color, LabColor)
    rgb_color = convert_color(lab_color, sRGBColor)
    if is_out_of_RGB_gamut(rgb_color):
        # filter out colors that are out of RGB gamut
        rgb_color = sRGBColor(0, 0, 0)
    return rgb_color

def get_hue_ranges(hue):
  hue_ranges = {
    'VR': [0, 24],
    'R': [24, 38],
    'RO': [38, 53],
    'O': [53, 65],
    'YO': [65, 80],
    'OY': [80, 90],
    'Y': [90, 100],
    'GY': [100, 115],
    'YG': [115, 130],
    'YG2': [130, 145],
    'G': [145, 162],
    'G2': [162, 180],
    'BG': [180, 204],
    'BG2': [204, 218],
    'GB': [218, 233],
    'GB2': [233, 245],
    'B': [245, 260],
    'B2': [260, 270],
    'VB': [270, 280],
    'VB2': [280, 295],
    'BV': [295, 310],
    'V': [310, 325],
    'RV': [325, 340],
    'VR': [340, 360],
  }    
  if hue in hue_ranges:
    return hue_ranges[hue] 
  else:
    return None   

def filter_colors_by_hue(lch_colors, hue): 
    hue_ranges = get_hue_ranges(hue)
    if hue_ranges:
        filtered_colors = []
        for lch_color in lch_colors:
            if lch_color.lch_h >= hue_ranges[0] and lch_color.lch_h < hue_ranges[1]:
                filtered_colors.append(lch_color)
        return filtered_colors
    else:
        return None   

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
        return 'YG2'
    elif h > 145 and h <= 162:
        return 'G'
    elif h > 162 and h <= 180:
        return 'G2'
    elif h > 180 and h <= 204:
        return 'BG'
    elif h > 204 and h <= 218:
        return 'BG2'
    elif h > 218 and h <= 233:
        return 'GB'
    elif h > 233 and h <= 245:
        return 'GB2'
    elif h > 245 and h <= 260:
        return 'B'
    elif h > 260 and h <= 270:
        return 'B2'
    elif h > 270 and h <= 280:
        return 'VB'
    elif h > 280 and h <= 295:
        return 'VB2'
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

def get_unique_l_values(rgb_colors):
    l_values = []
    for rgb_color in rgb_colors:
        lch_color = (rgb2lch(rgb_color[0], rgb_color[1], rgb_color[2]))
        l = round(lch_color.lch_l)
        if l not in l_values:
            l_values.append(l)
    l_values.sort(reverse=True)

    return l_values

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
    
def get_label_for_image(rgb_color):
    lch_color = rgb2lch(rgb_color.rgb_r, rgb_color.rgb_g, rgb_color.rgb_b)
    phrase = str(round(rgb_color.rgb_r)) + ',' + str(round(rgb_color.rgb_g)) + ',' + str(round(rgb_color.rgb_b)) + '\n'
    phrase += str(round(lch_color.lch_l)) + ',' + str(round(lch_color.lch_c)) + ',' + str(round(lch_color.lch_h))
    return phrase

def get_rgb_image_with_label(width, height, rgb_color):
    image_arr_rgb = generate_rgb_image_arr(width, height, rgb_color)
    image = Image.fromarray(image_arr_rgb).convert('RGBA')
    label = get_label_for_image(rgb_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Arial", 15)
    draw.text((20, 30), label, (255,255,255), font=font)
    return image

def get_rgb_image(width, height, rgb_color):
    image_arr_rgb = generate_rgb_image_arr(width, height, rgb_color)
    image = Image.fromarray(image_arr_rgb).convert('RGBA')
    return image    

def concat_images(images, size, shape=None):
    # Open images and resize them
    width, height = size
    # images = map(Image.open, image_paths)
    images = [ImageOps.fit(image, size, Image.ANTIALIAS)
              for image in images]
    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)
    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            image.paste(images[idx], offset)
    return image

def is_out_RGB_range(component):
    return component < 0 or component > 1

# TODO: check scaled or not
def is_out_of_RGB_gamut(rgb_color):
    if is_out_RGB_range(rgb_color.rgb_r) or is_out_RGB_range(rgb_color.rgb_g) or is_out_RGB_range(rgb_color.rgb_b):
        return True
    return False

def show_ch_swatch(chroma):
    hues = [15, 30, 45, 60, 70, 85, 95, 105, 120, 140, 150, 170, 190, 210, 225, 240, 250, 265, 275, 290, 300, 315, 330, 350]
    l_range = [0, 5, 10, 15, 25, 35, 40, 50, 60, 65, 75, 85, 90, 95, 100]
    rgb_colors = []

    for hue in hues:
        colors = []
        for l in l_range:
            lch_color = LCHabColor(l, chroma, hue)
            rgb_color = lch2rgb(lch_color.lch_l, lch_color.lch_c, lch_color.lch_h)
            colors.append(rgb_color)

        rgb_colors.append(colors)

    labeled_images = []
    for i in range(len(hues)):
        for j in range(len(l_range)):
            upscaled_rgb_color = sRGBColor(rgb_colors[i][j].rgb_r*255, rgb_colors[i][j].rgb_g*255, rgb_colors[i][j].rgb_b*255)
            labeled_images.append(get_rgb_image_with_label(100, 100,  upscaled_rgb_color))

    image = concat_images(labeled_images, (100, 100), (len(hues), len(l_range)))
    image.show()

def std_deviation_rgb(rgb_color):
    return np.std([rgb_color.rgb_r/255, rgb_color.rgb_g/255, rgb_color.rgb_b/255])

def chromatic_membership_degree(rgb_color):
    a = 0.1
    b = 0.2
    std_color = std_deviation_rgb(rgb_color)
    if std_color < a:
        return 0
    elif std_color >= a and std_color < (a+b)/2:
        return 2*pow((std_color-a)/(b-a), 2)
    elif std_color >= (a+b)/2 and std_color < b:
        return 1 - 2*pow((std_color-a)/(b-a), 2)
    elif std_color >= b:
        return 1