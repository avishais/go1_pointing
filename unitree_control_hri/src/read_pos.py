#!/usr/bin/python

import rospy
from scipy import empty
from geometry_msgs.msg import Twist
from std_msgs.msg import Header , Float64
from unitree_legged_msgs.msg import HighState
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
        self.SubTOHigh = rospy.Subscriber('/high_state',HighState,self.get_high_msg_msg)
        rospy.Subscriber('/unitree_keyboard_command', Float64, self.GetNextCommand )
        # self.OldPose = np.array([0,0,0])
        self.ChangeCommand = 0.0
        rospy.spin()

    def GetNextCommand(self,msg):
        self.ChangeCommand = msg.data
        
    def get_high_msg_msg(self,msg):
        # print("----------------------------------------------")
        # print(msg.position)
        if self.ChangeCommand == 8.0:
            self.startP =  msg.position
            self.ChangeCommand = 0
            print(msg.position,"start")
        if self.ChangeCommand == 5.0:
            print(msg.position,"end")
            print(np.array(self.startP) -np.array(msg.position),"delta")
            self.ChangeCommand = 0
        print(msg.position)


        # print(self.OldPose-msg.position)
        # self.OldPose = np.array(msg.position)
        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

if __name__ == '__main__':
    rospy.init_node('convetr')
    Convertor()

