from Processor import Processor
from Planner import Planner
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def main():
    img = cv2.imread('trumpet.jpeg')
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
    

    path = Planner.PathPlan(approx_contours, np.array([[0, 0]]))
    # drawn_path = DrawPath(img, path)
    # # Show the image with the line
    # cv2.imshow('Path Plan', drawn_path)
    # # Wait for a key press and close the window
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
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
    keys, values = DpThresholdToComplexity(img, low_bound, up_bound, num_points)
    plot_graph_without_dots(keys, values, 'Contour Approximation Threshold', 'Number of Points', 'Effect of Contour Approximation on Image Complexity')
    keys, values = DpThresholdToAccuracy(img, low_bound, up_bound, num_points)
    plot_graph_without_dots(keys, values, 'Contour Approximation Threshold', "Accuracy", 'Effect of Contour Approximation on Image Accuracy')

    low_bound = 0
    up_bound = 1000
    num_points = 50
    keys, values = EdgeDetectionThresholdToComplexity(img, low_bound, up_bound, num_points)
    plot_graph_without_dots(keys, values, 'Edge Detection Threshold', 'Number of Points', 'Effect of Edge Detection Threshold on Image Complexity')
    keys, values = EdgeDetectionThresholdToAccuracy(img, low_bound, up_bound, num_points)
    plot_graph_without_dots(keys, values, 'Edge Detection Threshold', "Accuracy", 'Effect of Edge Detection Threshold on Image Accuracy')

    keys, values = FilterContourThresholdToComplexity(img)
    plot_graph_without_dots(keys, values, 'Contour Similarity Threshold', 'Number of Points', 'Effect of Contour Filtering on Image Complexity', 'log')
    keys, values = FilterContourThresholdToAccuracy(img)
    plot_graph_without_dots(keys, values, 'Contour Similarity Threshold', 'Accuracy', 'Effect of Contour Filtering on Image Accuracy', 'log')


def show_img(img: np.ndarray) -> None:
    # Display the modified image
    cv2.imshow(f'Image (Press 0 to Exit)', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

# def plot_graph_with_dots(x, y, x_lable_str, title_str):

#     # Create the plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, y, marker='o', linestyle='-', color='b')

#     # # Set x-axis to integer scale
#     # plt.xticks(np.arange(min(x), max(x) + 1, step_val))  # Ensure all x ticks are integers

#     # Add titles and labels
#     plt.title(title_str)
#     plt.xlabel(x_lable_str)
#     plt.ylabel('Number of Points')

#     plt.grid()
#     plt.show()

def plot_graph_without_dots(x, y, x_label_str, y_label_str, title_str, scale=None):
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linestyle='-', color='b')  # Removed marker='o'

    # # Set x-axis to integer scale
    if scale:
        plt.xscale(scale)
    # plt.xticks(np.arange(min(x), max(x) + 1, step_val))  # Ensure all x ticks are integers

    # Add titles and labels
    plt.title(title_str)
    plt.xlabel(x_label_str)
    plt.ylabel(y_label_str)

    plt.grid()
    plt.show()


def DpThresholdToComplexity(img, low_bound, up_bound, num_points):
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)
    edge_img = Processor.EdgeDetection(blurred_img)
    contours = Processor.GetContours(edge_img)
    filtered_contours = Processor.FilterContours(contours, 0.5, 10)

    points = np.linspace(low_bound, up_bound, num_points)
    threshold_to_num_points = {}
    for p in points:
        approx_contours = Processor.ApproxContours(filtered_contours, p)
        threshold_to_num_points[p] = GetNumPoints(approx_contours)

    keys_array = list(threshold_to_num_points.keys())
    values_array = list(threshold_to_num_points.values())
    return keys_array, values_array

def EdgeDetectionThresholdToComplexity(img, low_bound, up_bound, num_points):
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)

    points = np.linspace(low_bound, up_bound, num_points)
    threshold_to_num_points = {}
    for p in points:
        edge_img = Processor.EdgeDetection(blurred_img, p)
        contours = Processor.GetContours(edge_img)
        filtered_contours = Processor.FilterContours(contours, 0.5, 10)
        approx_contours = Processor.ApproxContours(filtered_contours, 5)
        threshold_to_num_points[p] = GetNumPoints(approx_contours)

    keys_array = list(threshold_to_num_points.keys())
    values_array = list(threshold_to_num_points.values())
    return keys_array, values_array

def FilterContourThresholdToComplexity(img):
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)
    edge_img = Processor.EdgeDetection(blurred_img)
    contours = Processor.GetContours(edge_img)

    points = np.array([0.001, 0.01, 0.1, 1, 10])
    threshold_to_num_points = {}
    for p in points:
        filtered_contours = Processor.FilterContours(contours, p, 10)
        approx_contours = Processor.ApproxContours(filtered_contours, 5)
        threshold_to_num_points[p] = GetNumPoints(approx_contours)

    keys_array = list(threshold_to_num_points.keys())
    values_array = list(threshold_to_num_points.values())
    return keys_array, values_array

def DpThresholdToAccuracy(img, low_bound, up_bound, num_points):
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)
    edge_img = Processor.EdgeDetection(blurred_img)
    contours = Processor.GetContours(edge_img)
    filtered_contours = Processor.FilterContours(contours, 0.5, 10)

    points = np.linspace(low_bound, up_bound, num_points)
    y = []
    for p in points:
        approx_contours = Processor.ApproxContours(filtered_contours, p)
        path = Planner.PathPlan(approx_contours, np.array([[0, 0]]))
        canvas = DrawPath(img, path)
        score, _ = ssim(edge_img, canvas, full=True)
        y.append(score)

    return points, y

def EdgeDetectionThresholdToAccuracy(img, low_bound, up_bound, num_points):
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)

    points = np.linspace(low_bound, up_bound, num_points)
    y = []
    for p in points:
        edge_img = Processor.EdgeDetection(blurred_img, p)
        contours = Processor.GetContours(edge_img)
        filtered_contours = Processor.FilterContours(contours, 0.5, 10)
        approx_contours = Processor.ApproxContours(filtered_contours, 5)
        path = Planner.PathPlan(approx_contours, np.array([[0, 0]]))
        canvas = DrawPath(img, path)
        score, _ = ssim(edge_img, canvas, full=True)
        y.append(score)

    return points, y

def FilterContourThresholdToAccuracy(img):
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)
    edge_img = Processor.EdgeDetection(blurred_img)
    contours = Processor.GetContours(edge_img)

    points = np.array([0.001, 0.01, 0.1, 1, 10])
    y = []
    for p in points:
        filtered_contours = Processor.FilterContours(contours, p, 10)
        approx_contours = Processor.ApproxContours(filtered_contours, p)
        path = Planner.PathPlan(approx_contours, np.array([[0, 0]]))
        canvas = DrawPath(img, path)
        score, _ = ssim(edge_img, canvas, full=True)
        y.append(score)

    return points, y
    

def DrawPath(img, path):
    height, width, _ = img.shape
    canvas = np.zeros((height, width), dtype=np.uint8)

    for i in range(len(path) - 1):
        curr_x, curr_y = (path[i][0, 0], path[i][0, 1])
        next_x, next_y = (path[i + 1][0, 0], path[i + 1][0, 1])
        
        # Draw a black line between the points (BGR format: Blue, Green, Red)
        thickness = 1        # Thickness of the line
        cv2.line(canvas, (curr_x, curr_y), (next_x, next_y), 255, thickness)
    return canvas

def GetNumPoints(contours):
    num_points = 0
    for i in range (len(contours)):
        for _ in contours[i]:
            num_points += 1
    return num_points

if __name__ == '__main__':
    main() 