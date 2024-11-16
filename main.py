from ImageProcess.Processor import Processor
import numpy as np
import math
import cv2

def main():
    img = "img.png"
    processor = Processor(img)

    grayscale_img = processor.ColorToGrayscale()
    show_img(grayscale_img)

    blurred_img = processor.GaussianBlur(grayscale_img)
    show_img(blurred_img)

    


    



def show_img(img: np.ndarray) -> None:
    # Display the modified image
    cv2.imshow(f'Image (Press 0 to Exit)', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 