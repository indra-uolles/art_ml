def get_color_intensity(color):
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b


def get_color_intensity_2(color):
    r, g, b = color
    return (r + g + b) / 3