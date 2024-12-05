# #!/user/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
import math

class Artist:
  def __init__(self):
    rospy.init_node('turtlebot_artist', anonymous=True)

    self.velocity_publisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)

    # Set looping rate
    self.rate = rospy.Rate(10)

    self.vel_msg = Twist()

    self.curr_pos = np.array([[0, 0]])
    self.curr_angle = 0

  # def MoveTurtlebot(self):
  #   # Move forward for 5 seconds
  #   self.vel_msg.linear.x = 0.2
  #   self.vel_msg.angular.z = 0.0
  #   t0 = rospy.Time.now().to_sec()
  #   while (rospy.Time.now().to_sec() - t0) < 5:
  #     self.velocity_publisher.publish(self.vel_msg)
  #     self.rate.sleep()


  #   # Stop the robot
  #   self.vel_msg.linear.x = 0.0
  #   self.velocity_publisher.publish(self.vel_msg)
  #   rospy.sleep(1)
    
  #   # Rotate for 5 seconds
  #   self.vel_msg.angular.z = 0.5   # Rotate at 0.5 rad/s
  #   rospy.loginfo("Rotating")
  #   t0 = rospy.Time.now().to_sec()
  #   while (rospy.Time.now().to_sec() - t0) < 5:
  #     self.velocity_publisher.publish(self.vel_msg)
  #     self.rate.sleep()

  #   # Stop the robot
  #   self.vel_msg.angular.z = 0.0
  #   self.velocity_publisher.publish(self.vel_msg) 
  #   rospy.loginfo("Motion complete")


  def MoveForward(self, distance):
    rospy.loginfo("Info: Move Forward Started")
    self.vel_msg.linear.x = 0.1
    t0 = rospy.Time.now().to_sec()
    while (rospy.Time.now().to_sec() - t0) < distance:
      self.velocity_publisher.publish(self.vel_msg)
      self.rate.sleep()

    self.StopRobot()
    rospy.loginfo("Info: Move Forward Completed")


  def Rotate(self, angle_radian):
    rospy.loginfo("Info: Rotation Started")
    self.vel_msg.angular.z = angle_radian
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
    if angle < 0:
      angle += (2 * np.pi)

    # Do angle subtraction (in radians)
    return angle - self.curr_angle

  def Move(self, coords_final):
    distance = self.GetEuclidianDistance(self.curr_pos, coords_final)
    angle = self.GetRotationAngle(self.curr_pos, coords_final)
    self.Rotate(angle)
    self.MoveForward(distance)
    self.curr_pos = coords_final
    self.curr_angle += angle

if __name__ == '__main__':
  artist = Artist()
  # artist.MoveTurtlebot()

  # TEST
  # angle_r = artist.GetRotationAngle(np.array([[0, 0]]), np.array([[0, -2]]))
  # print("Radian: ", angle_r)

  # eucDist = artist.GetEuclidianDistance(np.array([[0, 0]]), np.array([[1, 1]]))
  # print("Distance: ", eucDist)

  square = np.array([np.array([[2, 0]]), np.array([[2, 2]]), np.array([[0, 2]]), np.array([[0, 0]])])
  triangle = np.array([np.array([[2, 0]]), np.array([[2, 2]]), np.array([[0, 0]])])

  for p in square:
    print(p)
    artist.Move(p)
  
  # for p in triangle:
  #   artist.Move(p)


# TODO
#  - figure out math for determining angle to turn given two coordinates
#  - function for calculating Euclidian distance given two coordinates
#  - store/keep track of robot current angle & position for future turning
#  - create test case to travel between pairs of coordinates