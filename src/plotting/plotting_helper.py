from matplotlib import colors

# determine transparent color equivalents
# https://stackoverflow.com/questions/33371939/calculate-rgb-equivalent-of-base-colors-with-alpha-of-0-5-over-white-background
def make_rgb_transparent(rgb, bg_rgb, alpha):
    return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, bg_rgb)]

def get_transparent_color(plot_object, bg_rgb=(1,1,1), alpha=0.2):
    color = plot_object[0].get_color() # get str value of color
    color = colors.colorConverter.to_rgb(color) # convert to tuple value of color
    color = make_rgb_transparent(color, bg_rgb, alpha)
    return color