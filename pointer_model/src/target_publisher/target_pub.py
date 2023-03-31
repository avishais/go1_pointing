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
from scipy.spatial.transform import Rotation


class target_pub(object):
    dog_height = 0.7 # meters
    Q_window = 1
    robot_pitch = np.deg2rad(0)
    def __init__(self):
        self.x, self.y, self.z, self.pitch, self.yaw = [], [], [], [], []
        rospy.Subscriber('NewPosition', Float64MultiArray, self.get_new_cords) 
        rospy.wait_for_message("NewPosition", Float64MultiArray)  
        self.publisher_ = rospy.Publisher('target', Float64MultiArray, queue_size=10)  
        self.listener = tf.TransformListener()

        self.Q = []
        while not rospy.is_shutdown():  
            try:
                (self.trans, self.rot) = self.listener.lookupTransform("camera_frame", "pointer", rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            self.compute_target()
            rospy.sleep(0.01)  


    def get_new_cords(self, msg):
        self.msg = msg
        # print(self.msg)
        self.x = self.msg.data[0]
        self.y = self.msg.data[1]
        self.z = self.msg.data[2]
        self.pitch = self.msg.data[4]
        self.yaw = self.msg.data[5]

    def compute_target(self):
        p = np.array([self.x, self.y, self.z])
        xh = self.get_pointing_vector()

        R = Rotation.from_quat(self.rot).as_dcm()
        Rrobot = Rotation.from_rotvec([0, self.robot_pitch, 0]).as_dcm() # Robot is inclined in angle 'robot_pitch'
        p = Rrobot.dot(p) # In robot frame
        xh = Rrobot.dot(xh)

        # Parametric equations
        t = (-self.dog_height-p[2])/xh[2]
        x_target = p[0] + t * xh[0]
        y_target = p[1] + t * xh[1]
        z_target = p[2] + t * xh[2]
        # xh_ = np.array([x_target, y_target, z_target]).T # In camera frame
        # xh = Rrobot.dot(xh_) # In robot frame

        self.Q.append(xh)
        if len(self.Q) > self.Q_window:
            del self.Q[0]

        target_p = Float64MultiArray()
        target_p.data = self.get_mean_target()   
        
        self.publisher_.publish(target_p)
        self.PubMarkerArrow("camera_frame", target_p.data, [0,0,0], [.0,1.0,0.0,1.0], 2, 0.15, 0.15)

    def get_pointing_vector(self):
        y = np.deg2rad(self.yaw)
        p = np.deg2rad(self.pitch)
        return np.array([np.cos(y)*np.cos(p), np.sin(y)*np.cos(p), -np.sin(p)]) # Updated to match the recorded coordinated frame of the camera.
    
    def get_mean_target(self):
        Q = np.array(self.Q)
        return list(np.mean(Q, axis = 0))
    
    def PubMarkerArrow(self,frame_id, pos, ori, color, marker_id, scaleX, scaleY):
        marker_pub = rospy.Publisher("/visualization_marker_target", Marker, queue_size = 2)

        marker = Marker()

        marker.header.frame_id = frame_id
        # marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 2
        marker.id = marker_id

        # Set the scale of the marker
        marker.scale.x = scaleX
        marker.scale.y = scaleY
        marker.scale.z = 0.15
        # marker.color = ColorRGBA([10.0, 2.0, 7.0, 0.8])
        # Set the color
        marker.color.r =color[0]
        marker.color.g =color[1]
        marker.color.b =color[2]
        marker.color.a =color[3]
        rot = Rotation.from_euler('xyz', ori, degrees=True)
            # Convert to quaternions and print
        rot_quat = rot.as_quat()
        # Set the pose of the marker
        marker.pose.position.x = float(pos[0])
        marker.pose.position.y = float(pos[1])
        marker.pose.position.z = float(pos[2])
        marker.pose.orientation.x = rot_quat[0]
        marker.pose.orientation.y = rot_quat[1]
        marker.pose.orientation.z = rot_quat[2]
        marker.pose.orientation.w = rot_quat[3]
        # rospy.sleep(0.5)
        # self.rate.sleep()
        marker_pub.publish(marker)

if __name__ == "__main__":
    rospy.init_node("target_pub")
    target_pub()
