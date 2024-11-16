import cv2
import numpy as np
from numpy.typing import NDArray
from skimage import color
from skimage import io


class Processor:
    def __init__(self, img: np.ndarray):
        self.img = img

    # TASKS
    # 1. color to grayscale
    # 2. apply gaussian blurr on grayscale
    # 3. extract edges with canny edge detection and apply 

    def ColorToGrayscale(self):
        grayscale_img = color.rgb2gray(io.imread(self.img))
        return grayscale_img
    
    def GaussianBlur(self, img):
        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
        return blurred_img
    
