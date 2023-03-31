#!/usr/bin/python

import rospy
from scipy import empty
from geometry_msgs.msg import Twist
from unitree_legged_msgs.msg import HighCmd
import numpy as np
# from visualization_msgs.msg import  Marker
# from std_msgs.msg import Float64MultiArray    
# import tf
# import math
# import tf
# import tf2_ros
# import geometry_msgs.msg

# from std_srvs.srv import Empty , EmptyResponse


class Convertor(object):
    def __init__(self):
        self.SubCmdVel = rospy.Subscriber('cmd_vel',Twist,self.get_twist_msg)
        self.PubHighCmd = rospy.Publisher('high_cmd',HighCmd,queue_size=1)
        # self.SubTOHigh = rospy.Subscriber('high_cmd',HighCmd,self.get_high_msg_msg)
        self.PubMsg = HighCmd()
        self.PubMsg.head = [254,239]
        self.PubMsg.levelFlag = 238

        self.PubMsg.mode = 0 # See example_walk
        self.PubMsg.gaitType = 1
        self.PubMsg.speedLevel = 0
        self.PubMsg.footRaiseHeight = 0
        self.PubMsg.bodyHeight = 0
        self.PubMsg.euler[0] = 0
        self.PubMsg.euler[1] = 0
        self.PubMsg.euler[2] = 0
        self.PubMsg.velocity[0] = 0.0
        self.PubMsg.velocity[1] = 0.0
        self.PubMsg.yawSpeed = 0.0
        self.PubMsg.reserve = 0
        rospy.spin()

    def get_high_msg_msg(self,msg):
        print(type(msg.head[0]))

    def get_twist_msg(self,msg):
        # print(msg)
        self.TwistMsg = msg
        self.PubMsg.mode = 2
        self.PubMsg.gaitType = 1
        self.PubMsg.velocity[0] = msg.linear.x
        self.PubMsg.velocity[1] = msg.linear.y
        self.yawSpeed = 0
        self.PubMsg.footRaiseHeight = 0.1
        # print(msg.linear.x,msg.linear.y)
        self.PubHighCmd.publish(self.PubMsg)
            

if __name__ == '__main__':
    rospy.init_node('convetr_cmd_vel_to_high_cmd_command_unitree')
    Convertor()

