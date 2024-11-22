import cv2
import numpy as np
from numpy.typing import NDArray
from skimage import color
from skimage import io
from typing import Tuple
from numpy.typing import NDArray

KERNAL_SIZE = 7
SIGMA = 0

# Particle info
RED = 180
GREEN = 105
BLUE = 255
PARTICLE_THICKNESS = 2
PARTICLE_RADIUS = 1

class Processor:
    def __init__(self):
        None

    def ColorToGrayscale(self, img):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayscale_img
    
    # Purpose: Apply Gaussian blur onto image
    def GaussianBlur(self, img):
        blurred_img = cv2.GaussianBlur(img, (KERNAL_SIZE, KERNAL_SIZE), SIGMA)
        return blurred_img
    
    # Purpose: Detect edges using Canny Edge Detection
    def EdgeDetection(self, img):
        edge_img = cv2.Canny(image=img, threshold1=50, threshold2=150)
        return edge_img    
    
    def GetContours(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        
        print("Number of Contours found = " + str(len(contours)))
        return contours

    # Purpose: Draw a particle at a location of the map assuming the given location is in bound of image
    def put_particle_at(self, coords, img: np.ndarray):
        copied_img = img.copy()

        # Particle appearence
        color = (BLUE, GREEN, RED)
        thickness = PARTICLE_THICKNESS

        # Draw particle on map
        cv2.circle(copied_img, coords, PARTICLE_RADIUS, color, thickness)
        return copied_img
