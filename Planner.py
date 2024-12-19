import numpy as np

class Planner:
    @staticmethod
    def PathPlan(contours, init_coord):
        path = np.array([init_coord])
        while len(contours) != 0:
          next_contour = Planner.GetNextContour(init_coord, contours)
          path = np.vstack((path, next_contour))
          init_coord = next_contour[0][0]

        return path

    # Given a coordinate pair and a list of contours, return the index of the contour and also the rearranged coordinates of this new contours
    @staticmethod
    def GetNextContour(coords, contours):
        closest_distance = np.linalg.norm(contours[0][0] - coords)
        closest_contour_idx = 0
        closest_coords_idx = 0
        for ci, c in enumerate(contours):
            for pi, p in enumerate(c):
                distance = np.linalg.norm(p - coords)
                if distance < closest_distance:
                    closest_coords_idx = pi
                    closest_distance = distance
                    closest_contour_idx = ci

        
        # rearrange coordinate
        rearranged_contour = np.roll(contours[closest_contour_idx], (-1 * closest_coords_idx), axis=0)
        del contours[closest_contour_idx]
        return np.append(rearranged_contour, rearranged_contour[0][np.newaxis, ...], axis=0)
