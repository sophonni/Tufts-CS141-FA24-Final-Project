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

    
    # Given a coordinate pair and a list of contours, return the index of the contour and also the rearranged coordinates of this new contours
    def Get_Next_Contour(coords, contours):
        # print(contours[0])
        closest_coords = contours[0][0]
        closest_distance = np.linalg.norm(closest_coords - coords)
        closest_contour = contours[0]
        # print("Closest Coord: ", closest_coords)
        # print("Closest Distance: ", closest_distance)
        # print("Closest Contour: ", closest_contour)
        for c in enumerate(contours):
            # print("here 1")
            for p in c:
                # print("here 2")
                distance = np.linalg.norm(p - coords)
                if distance < closest_distance:
                    closest_coords = p
                    closest_distance = distance
                    closest_contour = c
                    print("Closest Coord: ", closest_coords)
                    print("Closest Distance: ", closest_distance)
                    print("Closest Contour: ", closest_contour)
                # find Euclidian distance between coords and p
                # update curr_closest to p if Euclidian(p, c) < curr_closest
                # update the closest contour idx as well

        # rearrange coordinate

        coord_idx = np.where(closest_contour == closest_coords)
        # print(closest_contour)
        rearranged_contour = np.roll(closest_contour, (-1 * coord_idx))
        # print(rearranged_contour[0])
        return np.append(rearranged_contour, rearranged_contour[0][np.newaxis, ...], axis=0)
    # np.append(rearranged_contour, rearranged_contour[0], axis=0)