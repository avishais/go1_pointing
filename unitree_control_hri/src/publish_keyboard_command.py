#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point, Twist
from math import atan2
from std_msgs.msg import Header , Float64

class joystik_unitree(object):
    def __init__(self):
        rospy.init_node('control_gazebo_from_kebaord_opic')
        self.speed = Twist()
        # rospy.init_node('joystik_unitree', anonymous=True)
        rospy.Subscriber('/unitree_keyboard_command', Float64, self.GetNextCommand )
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
        rata = rospy.Rate(3)
        while  rospy.is_shutdown:
            self.pub.publish(self.speed)
            rata.sleep()    
        # rospy.spin()


    def GetNextCommand(self,msg):
        print(msg.data,"next Commend")
        if msg.data == 8.0:
            self.speed.linear.x = 0.5
            self.speed.angular.z = 0.0
        if msg.data == 2.0:
            self.speed.linear.x = -0.5
            self.speed.angular.z = 0.0

        if msg.data == 4.0:
            self.speed.linear.x = 0.5
            self.speed.angular.z = -0.3
        if msg.data == 6.0:
            self.speed.linear.x = -0.5
            self.speed.angular.z = 0.3

        if msg.data == 5.0:
            self.speed.linear.x = 0.0
            self.speed.angular.z = 0.0
        


if __name__ == '__main__':
    joystik_unitree()
   