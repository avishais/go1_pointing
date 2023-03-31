#! /usr/bin/env python3

import rospy
from custom_interfaces.msg import ModleData
import tf2_ros
import geometry_msgs.msg
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header , ColorRGBA
from scipy.spatial.transform import Rotation



class SimpleSubscriber(object):
    def __init__(self):
        rospy.init_node('simple_node')
        rospy.Subscriber('/PubModelData',ModleData,self.listener_callback)
        rospy.spin()
      

    def listener_callback(self, msg):
        self.get_logger().info('I receive: "%s"' % str(msg))
        self.PublishTF(msg)
        self.PubMarkerArrow(msg)

    def PublishTF(self,obj):
        self.br = tf2_ros.TransformBroadcaster(self)
        
        # broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        # static_transformStamped.header.stamp = rclpy.Time.now()
        static_transformStamped.header.frame_id = "camera_frame"
        static_transformStamped.child_frame_id = 'pointer'
        # print(obj.object_name,"obj.object_name")
        static_transformStamped.transform.translation.x = float(obj.x_cord)
        static_transformStamped.transform.translation.y = float(obj.y_cord)
        static_transformStamped.transform.translation.z = float(obj.z_cord)

        # quat = tf2_ros.transformations.quaternion_from_euler(            
        #         float(0),float(obj.pitch),float(obj.yaw))
        self.rot = Rotation.from_euler('xyz', [0, obj.pitch, obj.yaw], degrees=True)

        # Convert to quaternions and print
        self.rot_quat = self.rot.as_quat()
        # print(self.rot_quat)
        static_transformStamped.transform.rotation.x = self.rot_quat[0]
        static_transformStamped.transform.rotation.y = self.rot_quat[1]
        static_transformStamped.transform.rotation.z = self.rot_quat[2]
        static_transformStamped.transform.rotation.w = self.rot_quat[3]
        
        self.br.sendTransform(static_transformStamped)

    def PubMarkerArrow(self,TargetPose):
        marker_pub = self.create_publisher(Marker, "/visualization_marker", 10)
        # marker_pub = rospy.Publisher(, , queue_size = 2)

        marker = Marker()

        marker.header.frame_id = "camera_frame"
        # marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 0
        marker.id = 1

        # Set the scale of the marker
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        # marker.color = ColorRGBA([10.0, 2.0, 7.0, 0.8])
        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.5
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = TargetPose.z_cord
        marker.pose.position.y = TargetPose.x_cord
        marker.pose.position.z = TargetPose.y_cord
        marker.pose.orientation.x = self.rot_quat[0]
        marker.pose.orientation.y = self.rot_quat[1]
        marker.pose.orientation.z = self.rot_quat[2]
        marker.pose.orientation.w = self.rot_quat[3]
        # rospy.sleep(0.5)
        # self.rate.sleep()
        marker_pub.publish(marker)


if __name__ == '__main__':
    SimpleSubscriber()