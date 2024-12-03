import cv2
import numpy as np

GAUSS_KERNAL_SIZE = 3
GAUSS_SIGMA = 0

CONTRAST = 3
BRIGHTNESS = 50

# Particle info
RED = 180
GREEN = 105
BLUE = 255
PARTICLE_THICKNESS = 2
PARTICLE_RADIUS = 1

class Processor:
    def __init__(self):
        None

    def ColorToGrayscale(img):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayscale_img
    
    def AdjustContrast(img):
        adjusted_img = cv2.convertScaleAbs(img, alpha=CONTRAST, beta=BRIGHTNESS)
        return adjusted_img
    
    # Purpose: Apply Gaussian blur onto image
    def GaussianBlur(img):
        blurred_img = cv2.GaussianBlur(img, (GAUSS_KERNAL_SIZE, GAUSS_KERNAL_SIZE), GAUSS_SIGMA)
        return blurred_img
    
    # Purpose: Detect edges using Canny Edge Detection
    def EdgeDetection(img):
        edge_img = cv2.Canny(image=img, threshold1=50, threshold2=150)
        return edge_img    
    
    def GetContours(img):
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        
        print("Number of Contours found = " + str(len(contours)))
        return contours
    
    def FilterContours(contours, threshold, distance):
        filtered_contours = []

        for c1 in contours:
            if cv2.contourArea(c1) > 8:
                is_duplicate = False
                for c2 in filtered_contours:
                    match_value = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)
                    M1 = cv2.moments(c1)
                    M2 = cv2.moments(c2)
                    c1X = int(M1["m10"] / M1["m00"])
                    c1Y = int(M1["m01"] / M1["m00"])
                    c2X = int(M2["m10"] / M2["m00"])
                    c2Y = int(M2["m01"] / M2["m00"])
                    if match_value < threshold and abs(c2X - c1X) < distance and abs(c2Y - c1Y) < distance:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered_contours.append(c1)
        
        return filtered_contours
        
    def ApproxContours(contours, threshold):
        approx_contours = []

        for c in contours:
            approx_contours.append(cv2.approxPolyDP(c, threshold, True))
            
        return approx_contours


    # Purpose: Draw a particle at a location of the map assuming the given location is in bound of image
    def put_particle_at(coords, img: np.ndarray):
        copied_img = img.copy()

        # Particle appearence
        color = (BLUE, GREEN, RED)
        thickness = PARTICLE_THICKNESS

        # Draw particle on map
        cv2.circle(copied_img, coords, PARTICLE_RADIUS, color, thickness)
        return copied_img
