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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

class PubTfArrow(object):
    def __init__(self):
        rospy.Subscriber('PubModelData', ModleData,self.get_cords)
        rospy.Subscriber('NewPosition', Float64MultiArray,self.get_new_cords)  
        rospy.wait_for_message("PubModelData", ModleData)
        rospy.wait_for_message("NewPosition", Float64MultiArray)  
        self.listener = tf.TransformListener()
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.PublishTF("camera_frame_motive","original_pointer",[0,0,0],[0,0,0])
        while not rospy.is_shutdown():   
            self.PublishModelData()
            rospy.sleep(0.1)  

    def get_cords(self,msg):
        self.msg = msg
        self.x = self.msg.x_cord
        self.y = self.msg.y_cord
        self.z = self.msg.z_cord
        self.pitch = np.deg2rad(self.msg.pitch)
        self.yaw = np.deg2rad(-self.msg.yaw)
   
    def get_new_cords(self,msg):
        self.new_pos = msg.data
    
    def PublishModelData(self):          
        # self.PublishTF("camera_frame_motive","original_pointer",[self.x,self.y,self.z],[0, self.yaw, self.pitch])
        # self.PubMarkerArrow("original_pointer",[0,0,0],[0, 0, 0],[0.0,0.0,1.0,1.0],1,0.5,0.15)
        self.PublishTF("camera_frame", "pointer",[self.new_pos[0],self.new_pos[1],self.new_pos[2]],[np.deg2rad(self.new_pos[3]),np.deg2rad(self.new_pos[4]),np.deg2rad(self.new_pos[5])])
        self.PubMarkerArrow("pointer",[0,0,0],[0,0,0],[1.0,0.0,0.0,1.0],2,0.4,0.18)
        
    def PubMarkerArrow(self,frame_id,pos,ori,color,marker_id,scaleX,scaleY):
        # marker_pub = self.create_publisher(Marker, "/visualization_marker", 10)
        marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
        # marker_pub = rospy.Publisher(, , queue_size = 2)

        marker = Marker()

        marker.header.frame_id = frame_id
        # marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 0
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
    
    def PublishTF(self,frame_id,child_frame_id,pos,ori):
        
        static_transformStamped = geometry_msgs.msg.TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = frame_id
        # static_transformStamped.header.frame_id = "camera_link"
        static_transformStamped.child_frame_id = child_frame_id
        # print(obj.object_name,"obj.object_name")
        static_transformStamped.transform.translation.x = float(pos[0])
        static_transformStamped.transform.translation.y = float(pos[1])
        static_transformStamped.transform.translation.z = float(pos[2])
        quat = tf.transformations.quaternion_from_euler(
                float(ori[0]),float(ori[1]),float(ori[2]))
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]

        self.broadcaster.sendTransform(static_transformStamped)
    

if __name__ == "__main__":
    rospy.init_node("test_pointer_ros")
    PubTfArrow()