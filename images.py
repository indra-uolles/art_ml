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

def get_extended_list(list, element, size):
    for i in range(size):
        list.append(element)
    return list

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

def plot_cl(colors, color_name):
    for i in range(len(colors)):
        plt.scatter([colors[i][0]], [colors[i][1]], color=color_name)
    plt.show()

def show_hue_swatch(hue):
    rgb_colors = get_rgb_colors_arr([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 255])
    filtered_rgb_colors = filter_rgb_colors_by_hue(rgb_colors, hue)
    l_values = get_unique_l_values(filtered_rgb_colors)
    max_chroma_items = 0
    for l in l_values:
        chroma_items = len(filter_rgb_colors_by_lightness(filtered_rgb_colors, l, 0.05))
        if (chroma_items > max_chroma_items):
            max_chroma_items = chroma_items

    hue_colors = []

    for l in l_values:
        l_filtered_rgb_colors = filter_rgb_colors_by_lightness(filtered_rgb_colors, l, 0.05)

        if (len(l_filtered_rgb_colors) == 0):
            continue

        l_filtered_rgb_colors = sort_rgb_colors_by_chroma(l_filtered_rgb_colors)
        chroma_items = len(l_filtered_rgb_colors)
        delta = max_chroma_items - chroma_items
        
        if (delta > 0):
            l_filtered_rgb_colors = get_extended_list(l_filtered_rgb_colors, [0, 0, 0], delta)
            for i in range(delta):
                l_filtered_rgb_colors.append([0, 0, 0])
        
        hue_colors.append(l_filtered_rgb_colors)

    labeled_images = []
    for i in range(len(hue_colors)):
        for j in range(max_chroma_items):
            rgb_color = sRGBColor(hue_colors[i][j][0], hue_colors[i][j][1], hue_colors[i][j][2])
            labeled_images.append(get_rgb_image_with_label(100, 100, rgb_color))

    image = concat_images(labeled_images, (100, 100), (len(hue_colors), max_chroma_items))
    image.show()