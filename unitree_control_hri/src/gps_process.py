#!/usr/bin/python
import rospy
from geometry_msgs.msg import Point
import numpy as np
import matplotlib.pyplot as plt
import nvector as nv
import geopy.distance
from geographiclib.geodesic import Geodesic
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Float64MultiArray, Float64

class gps(object):
    heading = 0
    w = 10
    updtate_start = True
    start_acc = []

    def __init__(self):
        rospy.Subscriber('/gps', Point, self.get_gps_coor)
        rospy.wait_for_message("/gps", Point)  
        self.start_read = [self.gps_read.x, self.gps_read.y, self.gps_read.z]

        s = rospy.Service('gps_reset', Trigger, self.trigger)
        self.position_pub = rospy.Publisher('/robot_position', Float64MultiArray, queue_size=1)

        self.geod = Geodesic(6378388, 1/297.0) # the international ellipsoid
        msg_pos = Float64MultiArray()

        while not rospy.is_shutdown(): 
            print('Running gps process')
            if self.updtate_start:
                if len(self.start_acc) < self.w:
                    self.start_acc.append([self.gps_read.x, self.gps_read.y, self.gps_read.z])
                else:
                    self.start_read = np.mean(np.array(self.start_acc), axis=0)
                    self.updtate_start = False
                    self.start_acc = []
            else:
                self.update_xy()
                msg_pos.data = self.pos
                self.position_pub.publish(msg_pos)
            rospy.sleep(0.1) 

    def get_gps_coor(self, msg):
        self.gps_read = msg

    def get_bearing(self):
        lat1 = self.start_read[0]
        long1 = self.start_read[1]
        lat2 = self.gps_read.x
        long2 = self.gps_read.y
            
        brng = self.geod.Inverse(lat1, long1, lat2, long2)
        self.distance = brng['s12']
        self.azimuth = brng['azi1']
    
    def update_xy(self):
        self.get_bearing()

        az = self.azimuth - self.heading
        x = self.distance * np.cos(np.deg2rad(az))
        y = self.distance * np.sin(np.deg2rad(az))

        self.pos = np.array([x, y])

    def trigger(self, request):
        self.updtate_start = not self.updtate_start
        if self.updtate_start:
            self.start_acc = []
            print('Setting up zero location...')
            return TriggerResponse(success=True, message="Zero location set")
     



if __name__ == '__main__':
    rospy.init_node('gps_process')
    gps()
        