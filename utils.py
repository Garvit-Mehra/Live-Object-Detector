import numpy as np

# Manually define colors for specific classes
class_colors = {
    "person": (0, 0, 255),
    "car": (0, 255, 255),
    "dog": (255, 0, 255),
    # Add more classes and their respective colors here
}


def generate_color(class_id):
    """
    Generates a unique RGB color based on the class ID.
    If the class has a manually set color, it will use that.
    Otherwise, it generates a random color.
    """
    # Get the class name using class_id
    class_name = get_class_name_by_id(class_id)  # Function to get the class name by ID (to be implemented)

    # If the class has a manually set color, use it. Otherwise, generate a random color.
    return class_colors.get(class_name, generate_random_color(class_id))


def generate_random_color(class_id):
    """
    Generates a random RGB color for classes without a manually defined color.
    """
    np.random.seed(class_id)  # Use class_id as the seed for consistency
    color = np.random.randint(0, 256, 3).tolist()
    return tuple(color)


def get_class_name_by_id(class_id):
    """
    Returns the class name by its ID.
    This is a placeholder function, you should link this to your class list or file.
    For example, if using COCO dataset classes, you can use the classes file.
    """
    # This is a mock function. Replace with actual logic to fetch class names.
    # You could map the class ID to a name from a predefined list (e.g., from `coco.names`).
    classes = ["person", "bicycle", "car", "dog", "cat"]  # Example class names
    return classes[class_id]  # Assuming class_id is valid


def get_text_color(bg_color):
    """
    Determines whether black or white text is more readable on the given background color.
    Uses luminance formula: (0.299*R + 0.587*G + 0.114*B)
    """
    luminance = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)
