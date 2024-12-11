#!/user/bin/env python
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from threading import Thread, Event

import numpy as np
import math
import cv2

####################################################################################################################################
GAUSS_KERNAL_SIZE = 3
GAUSS_SIGMA = 0

CONTRAST = 3
BRIGHTNESS = 50

# Particle info
RED = 180
GREEN = 105
BLUE = 255
PARTICLE_THICKNESS = 2
PARTICLE_RADIUS = 1

DIST_SCALE = 0.03125
ANGLE_SCALE = 1.45

class Processor:
    @staticmethod
    def ColorToGrayscale(img):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayscale_img
    
    @staticmethod
    def AdjustContrast(img):
        adjusted_img = cv2.convertScaleAbs(img, alpha=CONTRAST, beta=BRIGHTNESS)
        return adjusted_img
    
    # Purpose: Apply Gaussian blur onto image
    @staticmethod
    def GaussianBlur(img):
        blurred_img = cv2.GaussianBlur(img, (GAUSS_KERNAL_SIZE, GAUSS_KERNAL_SIZE), GAUSS_SIGMA)
        return blurred_img
    
    # Purpose: Detect edges using Canny Edge Detection
    @staticmethod
    def EdgeDetection(img):
        edge_img = cv2.Canny(image=img, threshold1=50, threshold2=150)
        return edge_img    
    
    @staticmethod
    def GetContours(img):
        _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        return contours
    
    @staticmethod
    def FilterContours(contours, threshold, distance):
        filtered_contours = []

        for c1 in contours:
            if cv2.contourArea(c1) > 8:
                is_duplicate = False
                for c2 in filtered_contours:
                    match_value = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)
                    M1 = cv2.moments(c1)
                    M2 = cv2.moments(c2)
                    c1X = int(M1["m10"] / M1["m00"])
                    c1Y = int(M1["m01"] / M1["m00"])
                    c2X = int(M2["m10"] / M2["m00"])
                    c2Y = int(M2["m01"] / M2["m00"])
                    if match_value < threshold and abs(c2X - c1X) < distance and abs(c2Y - c1Y) < distance:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered_contours.append(c1)
        
        return filtered_contours
        
    @staticmethod
    def ApproxContours(contours, threshold):
        approx_contours = []

        for c in contours:
            approx_contours.append(cv2.approxPolyDP(c, threshold, True))
            
        return approx_contours
    
####################################################################################################################################
class Planner:
    @staticmethod
    def PathPlan(contours):
        curr_len = 0
        big_contour_idx = 0
        for idx, c in enumerate(contours):
            if c.shape[0] > curr_len:
                big_contour_idx = idx
        big_contour = contours[big_contour_idx]
        init_coord = big_contour[0][0]

        del contours[big_contour_idx]
        path = np.append(big_contour, big_contour[0][np.newaxis, ...], axis=0)
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

####################################################################################################################################
class Odometer:
    def __init__(self):
      self.pose = None
      self.running = True
    
    def run(self):
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        while (self.running and not rospy.is_shutdown()):
            rospy.sleep(0.1)  # Sleep to reduce CPU usage

    def odom_callback(self, msg):
       if not self.running:
          return
       self.pose = msg.pose
    
    def stop(self):
       self.running = False

####################################################################################################################################
class Artist:
  def __init__(self):
    rospy.init_node('turtlebot_artist', anonymous=True)

    self.velocity_publisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)

    # Set looping rate
    self.rate = rospy.Rate(10)

    self.vel_msg = Twist()

    self.curr_pos = np.array([[124, 34]])
    self.curr_angle = 0

    self.listener = Odometer()
    self.listener_thread = Thread(target=self.listener.run)
    self.listener_thread.start()

  def MoveForward(self, distance):
    rospy.loginfo("Info: Move Forward Started")
    self.vel_msg.linear.x = 0.2
    t0 = rospy.Time.now().to_sec()
    while (rospy.Time.now().to_sec() - t0) < distance * DIST_SCALE:
      self.velocity_publisher.publish(self.vel_msg)
      self.rate.sleep()

    self.StopRobot()
    rospy.loginfo("Info: Move Forward Completed")

  def Rotate(self, angle_radian):
    if angle_radian > np.pi:
       angle_radian = -((2 * np.pi) - angle_radian)
    elif angle_radian < (-1 * np.pi):
       angle_radian = ((2 * np.pi) + angle_radian)
       
    rospy.loginfo("Info: Rotation Started")
    rospy.loginfo("Info::Rotation::Curr Radian::%s", angle_radian)
    self.vel_msg.angular.z = angle_radian * ANGLE_SCALE
    t0 = rospy.Time.now().to_sec()
    while (rospy.Time.now().to_sec() - t0) < 1:
      self.velocity_publisher.publish(self.vel_msg)
      self.rate.sleep()

    self.StopRobot()
    rospy.loginfo("Info: Rotation Completed")
    

  def StopRobot(self):
    self.vel_msg.linear.x = 0.0
    self.vel_msg.angular.z = 0.0
    self.velocity_publisher.publish(self.vel_msg)
    rospy.sleep(1)

  def GetEuclidianDistance(self, coords_init, coords_final):
    distance = np.linalg.norm(coords_final - coords_init)
    return distance

  def GetRotationAngle(self, coords_init, coords_final):
    # Calculate the angle
    diff = coords_final - coords_init
    angle = np.arctan2(diff[0, 1], diff[0, 0])
    return angle - self.curr_angle

  def Move(self, coords_final):
    distance = self.GetEuclidianDistance(self.curr_pos, coords_final)
    angle = self.GetRotationAngle(self.curr_pos, coords_final)
    self.Rotate(angle)
    self.MoveForward(distance)
    self.curr_pos = coords_final
    self.curr_angle += angle
    print("Control estimated coords:", coords_final)
    print("Odometer pose:", self.listener.pose)

if __name__ == '__main__':
  artist = Artist()

  square = np.array([np.array([[2, 0]]), np.array([[2, 2]]), np.array([[0, 2]]), np.array([[0, 0]])])
  triangle = np.array([np.array([[4, 0]]), np.array([[4, 4]]), np.array([[0, 0]])])

  # for p in square:
  #   artist.Move(p)
  
  # for p in triangle:
  #   artist.Move(p)

  img = cv2.imread('amg.png')
  height, width, channels = img.shape

  grayscale_img = Processor.ColorToGrayscale(img)
  adjusted_img = Processor.AdjustContrast(grayscale_img)
  blurred_img = Processor.GaussianBlur(adjusted_img)
  edge_img = Processor.EdgeDetection(blurred_img)
  
  contours = Processor.GetContours(edge_img)
  filtered_contours = Processor.FilterContours(contours, 0.5, 10)
  approx_contours = Processor.ApproxContours(filtered_contours, 5)

  path = Planner.PathPlan(approx_contours)
  print("Odometer pose:", artist.listener.pose)
  for i in range(10):
    artist.Move(path[i])
  
  artist.listener.stop()
  artist.listener_thread.join()

  # c1 = Planner.GetNextContour(np.array([[124, 34]]), approx_contours)
  # print(c1)
  # for p in c1:
  #   print(p)
    # artist.Move(p)