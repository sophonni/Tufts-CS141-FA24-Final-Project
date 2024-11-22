from ImageProcess.Processor import Processor
import numpy as np
import math
import cv2

def main():
    img = cv2.imread('amg.png')
    processor = Processor()

    grayscale_img = processor.ColorToGrayscale(img)
    show_img(grayscale_img) 

    blurred_img = processor.GaussianBlur(grayscale_img)
    show_img(blurred_img)

    edge_img = processor.EdgeDetection(blurred_img)
    show_img(edge_img)

    # test tracing out detected edges
    particle_img = img
    height, width = np.shape(edge_img)

    for r in range(height):
        for c in range(width):
            if edge_img[r, c] == 255:
                coords = (c, r)
                particle_img = processor.put_particle_at(coords, particle_img)
    
    show_img(particle_img)
    

    contours = processor.GetContours(edge_img)
    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw the original contour (blue) and the approximated contour (green)
        cv2.drawContours(img, [contour], 0, (255, 0, 0), 2)
        cv2.imshow("Contour",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
        cv2.imshow("Contour",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    # print(contours[1])
    # print(np.shape(contours[1]))

    # ret,thresh=cv2.threshold(grayscale_img,200,255,cv2.THRESH_BINARY_INV)

    # countours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    cv2.drawContours(img,contours,-1,(0,255,0),3)
    cv2.imshow("Contour",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    



def show_img(img: np.ndarray) -> None:
    # Display the modified image
    cv2.imshow(f'Image (Press 0 to Exit)', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 