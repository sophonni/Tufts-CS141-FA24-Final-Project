import cv2
import numpy as np
from numpy.typing import NDArray
from skimage import color
from skimage import io

KERNAL_SIZE = 7
SIGMA = 0

class Processor:
    def __init__(self):
        None

    # TASKS
    # 1. color to grayscale
    # 2. apply gaussian blurr on grayscale
    # 3. extract edges with canny edge detection and apply 

    def ColorToGrayscale(self, img):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayscale_img
    
    def GaussianBlur(self, img):
        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(img, (KERNAL_SIZE, KERNAL_SIZE), SIGMA)
        return blurred_img
    
    def EdgeDetection(self, img):
        # Canny Edge Detection
        edges = cv2.Canny(image=img, threshold1=100, threshold2=200) # Canny Edge Detection
        return edges    
