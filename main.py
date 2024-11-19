from ImageProcess.Processor import Processor
import numpy as np
import math
import cv2

def main():
    img = cv2.imread('img.png')
    processor = Processor()

    grayscale_img = processor.ColorToGrayscale(img)
    show_img(grayscale_img)

    blurred_img = processor.GaussianBlur(grayscale_img)
    show_img(blurred_img)

    edge_img = processor.EdgeDetection(blurred_img)
    show_img(edge_img)

    edge_coordinates = processor.EdgeCoordinates(edge_img)

    # test tracing out detected edges
    particle_img = img
    show_img(particle_img)
    for coord in edge_coordinates:
        particle_img = processor.put_particle_at(coord, particle_img)
    
    show_img(particle_img)

    



def show_img(img: np.ndarray) -> None:
    # Display the modified image
    cv2.imshow(f'Image (Press 0 to Exit)', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 