import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute Idx and Idy with cv2.Sobel
    window_name = ('Haris Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    # gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    # Gradient-X
    Idx = cv2.Sobel(I, ddepth, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    Idy = cv2.Sobel(I, ddepth, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)

    # Step 2: Ixx Iyy Ixy from Idx and Idy
    Ixx = np.multiply(Idx, Idx)
    Iyy = np.multiply(Idy, Idy)
    Ixy = np.multiply(Idx, Idy)

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur
    # Use sdev = 1 and kernelSize = (3, 3) in cv2.GaussianBlur
    G = cv2.GaussianBlur(I, (3, 3), 1, sigmaY=1)
    A = Ixx * G
    B = Iyy * G
    C = Ixy * G

    # Step 4: Compute the harris response with the determinant and the trace of T
    # Method: 1
    # Wdet = Ixx * Iyy - Ixy ** 2
    # Wtr = Ixx + Iyy
    # R = Wdet / Wtr

    # Method 2
    Wdet = A * B - C ** 2
    Wtr = A + B
    # R = Wdet / Wtr
    R = Wdet - (k * np.square(Wtr))

    # # Method 3
    # T = np.array([A, C])
    # T = np.append(T, [C, B], axis=0)
    # R = np.linalg.det(T) - (k * np.trace(T) ^ 2)


    return R, A, B, C, Idx, Idy


def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization

    # Step 2 (recommended): Create one image for every offset in the 3x3 neighborhood

    # Step 3 (recommended): Compute the greatest neighbor of every pixel

    # Step 4 (recommended): Compute a boolean image with only all key-points set to True

    # Step 5 (recommended): Use np.nonzero to compute the locations of the key-points from the boolean image

    raise NotImplementedError


def detect_edges(R: np.array, edge_threshold: float = -0.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization

    # Step 2 (recommended): Calculate significant response pixels

    # Step 3 (recommended): Create two images with the smaller x-axis and y-axis neighbors respectively

    # Step 4 (recommended): Calculate pixels that are lower than either their x-axis or y-axis neighbors

    # Step 5 (recommended): Calculate valid edge pixels by combining significant and axis_minimal pixels

    raise NotImplementedError
