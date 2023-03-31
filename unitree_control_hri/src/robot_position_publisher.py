#!/usr/bin/python
import rospy
from unitree_legged_msgs.msg import HighState, IMU
from std_msgs.msg import Float64MultiArray, Float64
from geometry_msgs.msg import Pose2D
import numpy as np
import matplotlib.pyplot as plt
from std_srvs.srv import Trigger, TriggerResponse

class position_publisher(object):
    start_position = [0, 0, 0]
    current_odo_position = [0, 0, 0]
    current_gps_position = [0, 0]
    target = [0, 0, 0]
    yaw = 0.
    path_odo = []
    path_gps = []
    set_target = False

    def __init__(self):
        # rospy.Subscriber('/gps', Pose2D, self.get_gps_coor)
        rospy.Subscriber('/high_state', HighState, self.get_high_msg_msg)
        rospy.wait_for_message("/high_state", HighState)  
        rospy.Subscriber('/target', Float64MultiArray, self.get_target)
        rospy.wait_for_message("/target", Float64MultiArray)  
        rospy.Subscriber('/robot_position', Float64MultiArray, self.get_gps_position)
        rospy.wait_for_message("/robot_position", Float64MultiArray)  
        s = rospy.Service('set_target', Trigger, self.set_target)

        self.fig = plt.figure(0, figsize=(7, 4))

        self.position_pub = rospy.Publisher('/robot_position_odometry', Float64MultiArray, queue_size=1)
        self.yaw_pub = rospy.Publisher('/yaw', Float64, queue_size=1)

        # Should write a service that locks the target as desired.
        self.target_pub = rospy.Publisher('/target_new', Float64MultiArray, queue_size=1)
        self.start = True
        
        msg_pos = Float64MultiArray()
        msg_target = Float64MultiArray()
        msg_yaw = Float64()

        start = True
        while not rospy.is_shutdown(): 
            print('Running robot pub')
            if self.start and not np.all(self.start_position == 0):
                self.start_position = self.odo_position
                self.start = False

            self.current_odo_position = np.array(self.odo_position) - np.array(self.start_position)
            
            msg_pos.data = self.current_odo_position
            self.position_pub.publish(msg_pos)
            msg_yaw.data = self.yaw
            self.yaw_pub.publish(msg_yaw)

            if not self.set_target or start:
                self.target_new = np.array([[np.cos(self.yaw), -np.sin(self.yaw)], [np.sin(self.yaw), np.cos(self.yaw)]]).dot(self.target[:2])
                start = False
                
            msg_target.data = np.copy(self.target_new)
            self.target_pub.publish(msg_target)
            
            self.path_odo.append(self.current_odo_position)
            self.path_gps.append(self.current_gps_position)

            self.PlotModelData()
            rospy.sleep(0.1) 


    def get_high_msg_msg(self, msg):
        self.odo_position = msg.position

        self.IMU = msg.imu
        self.yaw = self.IMU.rpy[2]
        self.pitch = self.IMU.rpy[1]
        # float32[4] quaternion
        # float32[3] gyroscope
        # float32[3] accelerometer
        # float32[3] rpy
        # int8 temperature

    def get_target(self, msg):
        self.target = np.array(msg.data)

    def get_gps_position(self, msg):
        self.current_gps_pos = np.array(msg.data)

    def set_target(self, request):
        self.set_target = not self.set_target
        if self.set_target:
            return TriggerResponse(success=True, message="Setting current target!")
        else:
            return TriggerResponse(success=True, message="Releasing target!")

    def PlotModelData(self):  
        plt.cla()
        x_odo = self.current_odo_position[0]
        y_odo = self.current_odo_position[1]
        x_gps = self.current_gps_pos[0]
        y_gps = self.current_gps_pos[1]
        P_odo = np.array(self.path_odo)
        P_gps = np.array(self.path_gps)

        plt.plot(P_odo[:,0], P_odo[:,1], ':g')
        plt.plot(P_gps[:,0], P_gps[:,1], '-k')
        plt.plot(x_odo, y_odo, 'or')
        plt.plot([x_odo, x_odo + 0.05*np.cos(self.yaw)], [y_odo, y_odo + 0.05*np.sin(self.yaw)], '-b')
        plt.plot(x_gps, y_gps, 'og')
        plt.plot([x_gps, x_gps + 0.05*np.cos(self.yaw)], [y_gps, y_gps + 0.05*np.sin(self.yaw)], '-b')
        plt.plot(self.target_new[0], self.target_new[1], 'dr', markersize = 5)
             
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Odometry position: (%.3f,%.3f)m' % (x_odo, y_odo))
        plt.title('GPS position: (%.3f,%.3f)m' % (x_gps, y_gps))
        plt.axis('equal')
        plt.draw()
        plt.pause(0.01)  


if __name__ == '__main__':
    rospy.init_node('robot_position')
    position_publisher()