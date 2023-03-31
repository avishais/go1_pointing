#!/usr/bin/env python
# license removed for brevity
import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Header , Float64
from math import atan2



class KeyBoardJoystickHusky(object):
    def __init__(self):
        self.pub = rospy.Publisher("/unitree_keyboard_command", Float64, queue_size = 1)
        initielizeNum = 0
        while not rospy.is_shutdown():
            if initielizeNum == 0:
                self.PrintInstraction()
                initielizeNum = 1
            num = int(input ("Enter Command :"))
            while num != 8 and num !=6 and num !=2 and num!=4 and num!=5:
                print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                print("Not Valid Inpud Pleas Enter New Command")
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                self.PrintInstraction()
                num = int(input ("Enter Command :"))
            # print(num, type(num))
            nextCommand = Float64()
            nextCommand.data = num
            self.pub.publish(nextCommand)
    
    def PrintInstraction(self):
        print("------------------------------")
        print("Forward :8")
        print("Backwards :2")
        print("Right: 6")
        print("Left: 4")
        print("Stop : 5")
        print("------------------------------")


if __name__ == '__main__':
    rospy.init_node('keyboard_command_topic_node')
    KeyBoardJoystickHusky()