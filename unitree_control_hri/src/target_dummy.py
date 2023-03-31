#!/usr/bin/python
import rospy
from pointer_model.msg import ModleData
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
import tf
import tf2_ros
import geometry_msgs.msg
from visualization_msgs.msg import Marker
import numpy as np
import math


class target_pub(object):
    def __init__(self):
        self.publisher_ = rospy.Publisher('/target', Float64MultiArray, queue_size=10)  

        self.Q = []
        while not rospy.is_shutdown():  

            target_p = Float64MultiArray()
            target_p.data = [1.3, 0.9, 0]  
        
            self.publisher_.publish(target_p)
            rospy.sleep(0.1)  

if __name__ == "__main__":
    rospy.init_node("target_pub")
    target_pub()
