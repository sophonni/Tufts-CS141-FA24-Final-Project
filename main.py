from Processor import Processor
from Planner import Planner
import numpy as np
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
    # for contour in approx_contours:
    #     print(contour.shape)
    #     cv2.drawContours(canvas, [contour], 0, (0, 255, 0), 2)
    #     cv2.imshow("Contour",canvas)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    

    path = Planner.PathPlan(approx_contours)
    # Create a blank white image (or you can load an existing image)
    canvas = np.ones((height, width, channels)) * 255

    for i in range(len(path) - 1):
        curr_x, curr_y = (path[i][0, 0], path[i][0, 1])
        next_x, next_y = (path[i + 1][0, 0], path[i + 1][0, 1])
        
        # Draw a black line between the points (BGR format: Blue, Green, Red)
        color = (0, 0, 0)  # Red color in BGR
        thickness = 2        # Thickness of the line
        cv2.line(canvas, (curr_x, curr_y), (next_x, next_y), color, thickness)

    # Show the image with the line
    cv2.imshow('Path Plan', canvas)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_img(img: np.ndarray) -> None:
    # Display the modified image
    cv2.imshow(f'Image (Press 0 to Exit)', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 