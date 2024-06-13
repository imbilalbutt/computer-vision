import numpy as np
from typing import List, Tuple
import cv2
import os

t_image_list = List[np.array]
t_str_list = List[str]
t_image_triplet = Tuple[np.array, np.array, np.array]


def show_images(images: t_image_list, names: t_str_list) -> None:
    """Shows one or more images at once.

    Displaying a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        names: A list of strings that will appear as the window titles for each image

    Returns:
        None
    """
    for i, image in enumerate(images):
        cv2.imshow(names[i], image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_images(images: t_image_list, filenames: t_str_list, **kwargs) -> None:
    """Saves one or more images at once.

    Saving a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        filenames: A list of strings where each respective file will be created

    Returns:
        None
    """
    for i, image in enumerate(images):
        cv2.imwrite(filenames[i], image)


def scale_down(image: np.array) -> np.array:
    """Returns an image half the size of the original.

    Args:
        image: A numpy array with an opencv image

    Returns:
        A numpy array with an opencv image half the size of the original image
    """
    resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    return resized


def separate_channels(colored_image: np.array) -> t_image_triplet:
    """Takes an BGR color image and splits it three images.

    Args:
        colored_image: an numpy array sized [HxWxC] where the channels are in BGR (Blue, Green, Red) order

    Returns:
        A tuple with three BGR images the first one containing only the Blue channel active, the second one only the
        green, and the third one only the red.
    """
    # colored_image[:, 1, 1] = 0
    blue, green, red = cv2.split(colored_image)

    # Create a zero matrix for the blue channel
    zeros = np.zeros_like(blue)

    blue_i = cv2.merge([blue, zeros, zeros])
    green_i = cv2.merge([zeros, green, zeros])
    red_i = cv2.merge([zeros, zeros, red])

    return (blue_i, green_i, red_i)
