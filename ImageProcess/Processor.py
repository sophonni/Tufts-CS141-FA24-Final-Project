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
PARTICLE_THICKNESS = 3
PARTICLE_RADIUS = 3

class Processor:
    def __init__(self):
        None

    def ColorToGrayscale(self, img):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayscale_img
    
    def GaussianBlur(self, img):
        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(img, (KERNAL_SIZE, KERNAL_SIZE), SIGMA)
        return blurred_img
    
    def EdgeDetection(self, img):
        # Canny Edge Detection
        edge_img = cv2.Canny(image=img, threshold1=100, threshold2=200)
        return edge_img    

    def EdgeCoordinates(self, img):
        # Get location of all edges represented by a 1x2 array
        edge_coordinates1x2 = np.column_stack(np.where(img > 0))

        # Initialize an empty array to store locations of all edges, with a shape of (0, 2) initially
        edge_coordinates2x1 = []
        for coord in edge_coordinates1x2:
            x = coord[0]
            y = coord[1]
            vector = np.array([[x], [y]])
            edge_coordinates2x1.append(vector)

        return edge_coordinates2x1

    # Purpose: Draw a particle at a location of the map assuming the given location is in bound of image
    def put_particle_at(self, pos_vec, img: np.ndarray):
        copied_img = img.copy()

        x_coord, y_coord = self.vec_to_coord(pos_vec)

        # Ensure pixel pos is not decimal
        x_coord = int(x_coord)
        y_coord = int(y_coord)

        # Particle appearence
        color = (BLUE, GREEN, RED)
        thickness = PARTICLE_THICKNESS

        # Draw particle on map
        cv2.circle(copied_img, (x_coord, y_coord), PARTICLE_RADIUS, color, thickness)
        return copied_img

    def vec_to_coord(self, state_vec: NDArray[np.double]) -> Tuple[int, int]:
        x_coord = state_vec[0, 0]
        y_coord = state_vec[1, 0]
        return x_coord, y_coord