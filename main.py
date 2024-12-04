from ImageProcess.Processor import Processor
from PathPlan.Planner import Planner
import numpy as np
import math
import cv2

def main():
    img = cv2.imread('amg.png')
    height, width, channels = img.shape
    canvas = np.ones((height, width, channels)) * 255

    grayscale_img = Processor.ColorToGrayscale(img)
    show_img(grayscale_img) 

    adjusted_img = Processor.AdjustContrast(grayscale_img)
    show_img(adjusted_img)

    blurred_img = Processor.GaussianBlur(adjusted_img)
    show_img(blurred_img)

    edge_img = Processor.EdgeDetection(blurred_img)
    show_img(edge_img)
    
    # Test: See Drawing
    contours = Processor.GetContours(edge_img)
    filtered_contours = Processor.FilterContours(contours, 0.5, 10)
    approx_contours = Processor.ApproxContours(filtered_contours, 5)
    for contour in approx_contours:
        print(contour.shape)
        cv2.drawContours(canvas, [contour], 0, (0, 255, 0), 2)
        cv2.imshow("Contour",canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # print(approx_contours[0][0])

    # Test: Find Next Contour
    # start_coord = np.array([[55, 169]])
    # start_coord = np.array([[147, 140]])

    # start_coord = np.array([[100, 100]])
    # print(Planner.GetNextContour(start_coord, approx_contours))
    print(Planner.PathPlan(approx_contours))


def show_img(img: np.ndarray) -> None:
    # Display the modified image
    cv2.imshow(f'Image (Press 0 to Exit)', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 