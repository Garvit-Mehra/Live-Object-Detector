import numpy as np


def generate_color(class_id):
    """
    Generates a unique RGB color based on the class ID.
    """
    np.random.seed(class_id)
    color = np.random.randint(0, 256, 3).tolist()
    return tuple(color)


def get_text_color(bg_color):
    """
    Determines whether black or white text is more readable on the given background color.
    Uses luminance formula: (0.299*R + 0.587*G + 0.114*B)
    """
    luminance = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)
