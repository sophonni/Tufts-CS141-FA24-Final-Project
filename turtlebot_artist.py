#!/user/bin/env python

import rospy
from geometry_msgs.msg import Twist

class Artist:
  def __init__(self):
    rospy.init_node('turtlebot_artist', anonymous=True)

    self.velocity_publisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)

    # Set looping rate
    self.rate = rospy.Rate(10)

    self.vel_msg = Twist()

  def move_turtlebot(self):
    # Move forward for 5 seconds
    self.vel_msg.linear.x = 0.2
    self.vel_msg.angular.z = 0.0
    t0 = rospy.Time.now().to_sec()
    while (rospy.Time.now().to_sec() - t0) < 5:
      self.velocity_publisher.publish(self.vel_msg)
      self.rate.sleep()


    # Stop the robot
    self.vel_msg.linear.x = 0.0
    self.velocity_publisher.publish(self.vel_msg)
    rospy.sleep(1)
    
    # Rotate for 5 seconds
    self.vel_msg.angular.z = 0.5   # Rotate at 0.5 rad/s
    rospy.loginfo("Rotating")
    t0 = rospy.Time.now().to_sec()
    while (rospy.Time.now().to_sec() - t0) < 5:
      self.velocity_publisher.publish(self.vel_msg)
      self.rate.sleep()

    # Stop the robot
    self.vel_msg.angular.z = 0.0
    self.velocity_publisher.publish(self.vel_msg) 
    rospy.loginfo("Motion complete")

  def move_forward(self, distance):
    rospy.loginfo("Info: Move Forward Started")
    self.vel_msg.linear.x = 0.1
    t0 = rospy.Time.now().to_sec()
    while (rospy.Time.now().to_sec() - t0) < distance:
      self.velocity_publisher.publish(self.vel_msg)
      self.rate.sleep()

    self.stop_robot()
    rospy.loginfo("Info: Move Forward Completed")

  
  def rotate(self, angle):
    self.vel_msg.linear.x = 0.1
    t0 = rospy.Time.now().to_sec()
    while (rospy.Time.now().to_sec() - t0) < distance:
      self.velocity_publisher.publish(self.vel_msg)
      self.rate.sleep()

    self.stop_robot()
    
  def stop_robot(self):
    self.vel_msg.linear.x = 0.0
    self.vel_msg.angular.z = 0.0
    self.velocity_publisher.publish(self.vel_msg)
    rospy.sleep(1)


if __name__ == '__main__':
  artist = Artist()
  artist.move_turtlebot()



# TODO
#  - figure out math for determining angle to turn given two coordinates
#  - function for calculating Euclidian distance given two coordinates
#  - store/keep track of robot current angle & position for future turning
#  - create test case to travel between pairs of coordinates