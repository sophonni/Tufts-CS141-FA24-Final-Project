from Processor import Processor
from Planner import Planner
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    img = cv2.imread('amg.png')
    height, width, channels = img.shape
    canvas = np.ones((height, width, channels)) * 255

    grayscale_img = Processor.ColorToGrayscale(img)
    # show_img(grayscale_img) 

    adjusted_img = Processor.AdjustContrast(grayscale_img)
    # show_img(adjusted_img)

    blurred_img = Processor.GaussianBlur(adjusted_img)
    # show_img(blurred_img)

    edge_img = Processor.EdgeDetection(blurred_img)
    # show_img(edge_img)
    
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
    

    # path = Planner.PathPlan(approx_contours, np.array([[0, 0]]))
    # # print(path)
    # # Create a blank white image (or you can load an existing image)
    # canvas = np.ones((height, width, channels)) * 255

    # for i in range(len(path) - 1):
    #     curr_x, curr_y = (path[i][0, 0], path[i][0, 1])
    #     next_x, next_y = (path[i + 1][0, 0], path[i + 1][0, 1])
        
    #     # Draw a black line between the points (BGR format: Blue, Green, Red)
    #     color = (0, 0, 0)  # Red color in BGR
    #     thickness = 2        # Thickness of the line
    #     cv2.line(canvas, (curr_x, curr_y), (next_x, next_y), color, thickness)

    # # Show the image with the line
    # cv2.imshow('Path Plan', canvas)

    # # Wait for a key press and close the window
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    low_bound = 0
    up_bound = 10
    num_points = 100
    keys, values = DpThreasholdToComplexity(img, low_bound, up_bound, num_points)
    plot_graph_without_dots(keys, values, 'Douglas-Peucker (DP) Threashold', f'Graph of {num_points} Different Threashold Value ranging from {low_bound} to {up_bound}')

    keys, values = EdgeDetectionThreasholdToComplexity(img, 0, 1000, 50)
    plot_graph_without_dots(keys, values, 'Edge Detection Threashold1', f'Graph of {num_points} Different Threashold Value ranging from {low_bound} to {up_bound}')


def show_img(img: np.ndarray) -> None:
    # Display the modified image
    cv2.imshow(f'Image (Press 0 to Exit)', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

def plot_graph_with_dots(x, y, x_lable_str, title_str):

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')

    # # Set x-axis to integer scale
    # plt.xticks(np.arange(min(x), max(x) + 1, step_val))  # Ensure all x ticks are integers

    # Add titles and labels
    plt.title(title_str)
    plt.xlabel(x_lable_str)
    plt.ylabel('Number of Points')

    plt.grid()
    plt.show()

def plot_graph_without_dots(x, y, x_label_str, title_str):
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linestyle='-', color='b')  # Removed marker='o'

    # # Set x-axis to integer scale
    # plt.xticks(np.arange(min(x), max(x) + 1, step_val))  # Ensure all x ticks are integers

    # Add titles and labels
    plt.title(title_str)
    plt.xlabel(x_label_str)
    plt.ylabel('Number of Points')

    plt.grid()
    plt.show()


def DpThreasholdToComplexity(img, low_bound, up_bound, num_points):
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)
    edge_img = Processor.EdgeDetection(blurred_img)
    contours = Processor.GetContours(edge_img)
    filtered_contours = Processor.FilterContours(contours, 0.5, 10)

    points = np.linspace(low_bound, up_bound, num_points)
    threashold_to_num_points = {}
    for p in points:
        approx_contours = Processor.ApproxContours(filtered_contours, p)
        threashold_to_num_points[p] = GetNumPoints(approx_contours)

    keys_array = list(threashold_to_num_points.keys())
    values_array = list(threashold_to_num_points.values())
    return keys_array, values_array

def EdgeDetectionThreasholdToComplexity(img, low_bound, up_bound, num_points):
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)

    points = np.linspace(low_bound, up_bound, num_points)
    threashold_to_num_points = {}
    for p in points:
        edge_img = Processor.EdgeDetection(blurred_img, p)
        contours = Processor.GetContours(edge_img)
        filtered_contours = Processor.FilterContours(contours, 0.5, 10)
        approx_contours = Processor.ApproxContours(filtered_contours, 5)
        threashold_to_num_points[p] = GetNumPoints(approx_contours)

    keys_array = list(threashold_to_num_points.keys())
    values_array = list(threashold_to_num_points.values())
    return keys_array, values_array
    


def GetNumPoints(contours):
    num_points = 0
    for i in range (len(contours)):
        for p in contours[i]:
            num_points += 1
    return num_points

if __name__ == '__main__':
    main() 