import cv2
import numpy as np

class Planner:
    def __init__(self):
        pass

    # Goal: Given a list of contours:
    # - Identify the biggest contour (most points)
    # - Pick a coordinate from the biggest contour (most points) as a start point
    # - Finish tracing out the contour
    # - Look through all the other contours and find a point that is closest to the last point in the current contour
    # - Mark current contour as visited
    # - trace out the new contour
    # - REPEAT

    # Given a list of contours
    # Create a new list path
    # path = biggest contour in the list of contours
    # remove the biggest contour from the list of contours
    # while the list of contours isn't empty:
        # Take the coordinate we're currently at, compare it to all the points in all the contours
        # Choose the closest point, add that contour point by point to path, then we delete the contour

    def PathPlan(contours):
        # currLen = 0
        # bigContourIdx = 0
        # for idx, c in enumerate(contours):
        #     if len(c[1]) > currLen:
        #         bigContourIdx 

        longest_idx = np.argmax([c[1].shape[0] for c in contours])
        path = contours[longest_idx]
        np.delete(contours, longest_idx)
        print("Contours:")

        return path

    # Given a coordinate pair and a list of contours, return the index of the contour and also the rearranged coordinates of this new contours
    def GetNextContour(coords, contours):
        closest_distance = np.linalg.norm(contours[0][0] - coords)
        closest_contour = contours[0]
        closest_coords_idx = 0
        for c in enumerate(contours):
            for idx, p in enumerate(c[1]):
                distance = np.linalg.norm(p - coords)
                if distance < closest_distance:
                    closest_coords_idx = idx
                    closest_distance = distance
                    closest_contour = c[1]

        # rearrange coordinate
        rearranged_contour = np.roll(closest_contour, (-1 * closest_coords_idx), axis=0)
        return np.append(rearranged_contour, rearranged_contour[0][np.newaxis, ...], axis=0)
