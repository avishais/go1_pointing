#!/usr/bin/env /home/roblab21/catkin_ws/src/pointer_model/src/melodic_py3/bin/python3
import rospy
from pointer_model.msg import ModleData
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
import geometry_msgs.msg
from visualization_msgs.msg import Marker
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

class MathPlotLibCheck(object):
    def __init__(self):
        rospy.Subscriber('PubPointRecored', ModleData,self.get_cords)
        self.publisher_ = rospy.Publisher('NewPosition', Float64MultiArray,queue_size=10)  
        self.fig = plt.figure(2, figsize=(10,7))
        self.ax = plt.axes(projection='3d')
        self.scale = 0.5
        self.ax.view_init(elev=16., azim=43)
        self.x  = 0
        self.y  = 0
        self.z  = 0
        self.pitch  = 0
        self.yaw  = 0
        self.xL = []
        self.yL = []
        self.zL = []
        self.New_POSITIONS = []
        self.CreatRotetionMatrixList()
        while not rospy.is_shutdown():   
            # print(scipy.__version__)  
            # plt.draw()
            rospy.wait_for_message("PubPointRecored", ModleData)
            self.PublishModelData()
                     
    def get_cords(self,msg):
        self.msg = msg
   
    def PublishModelData(self):  
        plt.cla()
        self.x = self.msg.x_cord
        self.y = self.msg.y_cord
        self.z = self.msg.z_cord
        self.pitch = self.msg.pitch
        self.yaw = self.msg.yaw
             
        self.arrowSizeX = 0.15
        self.arrowSizeY = 0.15
        self.arrowSizeZ = 0.15
        self.PlotAxis()        
        self.PlotOriginalRecorsedData()
        # self.PublishTF("camera_frame_motive","original_pointer",[self.x,self.y,self.z],[0, self.yaw, self.pitch])
        # self.PubMarkerArrow("original_pointer",[0,0,0],[0, 0, 0],[0.0,0.0,1.0,1.0],1,0.5,0.15)
        self.PlotTransRecordedData() 
        new_p = Float64MultiArray()
        new_p.data = [self.new_position[0],self.new_position[1],self.new_position[2],self.new_euler[0],-self.new_euler[1],-self.new_euler[2]]
        self.publisher_.publish(new_p)
        # self.PublishTF("camera_frame","pointer",[self.new_position[0],self.new_position[1],self.new_position[2]],[self.new_euler[0],-self.new_euler[1],-self.new_euler[2]])
        # self.PubMarkerArrow("pointer",[0,0,0],[0,0,0],[1.0,0.0,0.0,1.0],2,0.4,0.18)
        
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_title('Position: (%.3f,%.3f,%.3f)m'%(self.new_position[0], self.new_position[1], self.new_position[2]) + ', pitch: %.2fdeg'%self.pitch + ', yaw: %.2fdeg'%(180+self.yaw))
        self.set_axes_equal()
        plt.draw()
        plt.pause(0.01)     
    
    def Rot(self,x):
        self.Rzx = Rotation.from_euler('zyx', [0, 180, 90], degrees=True)
        return np.array(self.Rzx.as_matrix().dot(x))

    def PlotTransRecordedData(self):
        TransformRecordedData = []
        rotetionM = self.T(self.map[0],self.map[1])
        TransformRecordedData.append(rotetionM)
        rotetionM = np.dot(rotetionM,self.T(self.camera_frame_motive[0],self.camera_frame_motive[1]))
        TransformRecordedData.append(rotetionM)
        rotetionM = np.dot(rotetionM,self.T(self.camera_frame[0],self.camera_frame[1]))
        TransformRecordedData.append(rotetionM)
        self.new_position = self.Rot([self.x,self.y,self.z])
       
        v = np.array([-np.cos(np.deg2rad(180-self.yaw))*np.cos(-np.deg2rad(self.pitch)), np.sin(np.deg2rad(180-self.yaw))*np.cos(-np.deg2rad(self.pitch)), -np.sin(-np.deg2rad(self.pitch))])
        rotetionX = np.asarray([[1, 0, 0], [0,math.cos(self.yaw),-math.sin(self.yaw)], [0,math.sin(self.yaw),math.cos(self.yaw)]])
        rotetionZ = np.asarray([[math.cos(np.deg2rad(180-self.yaw)), -math.sin(np.deg2rad(180-self.yaw)) ,0], [math.sin(np.deg2rad(180-self.yaw)),math.cos(np.deg2rad(180-self.yaw)), 0], [0, 0, 1]])
        rotetionY = np.asarray([[math.cos(-np.deg2rad(self.pitch)), 0 ,math.sin(-np.deg2rad(self.pitch))], [0,1, 0], [-math.sin(-np.deg2rad(self.pitch)), 0, math.cos(-np.deg2rad(self.pitch))]])
        self.Rotetionxz = Rotation.from_matrix(np.dot(rotetionY,rotetionZ))
        
        self.new_euler = self.Rotetionxz.as_euler('xyz',degrees=True)
        rotetionM = np.dot(rotetionM,self.T([self.new_position[0],self.new_position[1],self.new_position[2]],[self.new_euler[0],self.new_euler[1],-self.new_euler[2]]))
        TransformRecordedData.append(rotetionM)
        self.rotetionL = []
        self.PoseL = []
        for rotetion in TransformRecordedData:
            self.rotetionL.append(np.array([rotetion[0][:-1],rotetion[1][:-1],rotetion[2][:-1]]))
            self.PoseL.append(np.array([rotetion[0][-1],rotetion[1][-1],rotetion[2][-1]]))
        self.PoseL = np.array(self.PoseL)
        self.ax.quiver(self.PoseL[-1][0],self.PoseL[-1][1],self.PoseL[-1][2],0.45*self.rotetionL[-1][0][0],0.45*self.rotetionL[-1][1][0],0.45*self.rotetionL[-1][2][0],color = 'red',linewidth=1.5)
        self.ax.quiver(self.PoseL[-1][0],self.PoseL[-1][1],self.PoseL[-1][2],0.6*v[0],0.6*v[1],0.6*v[2],color = 'green',linewidth=1.5)
        
    def PlotOriginalRecorsedData(self):
        OriginalRecordedData = []
        rotetionM = self.T(self.map[0],self.map[1])
        OriginalRecordedData.append(rotetionM)
        rotetionM = np.dot(rotetionM,self.T(self.camera_frame_motive[0],self.camera_frame_motive[1]))
        OriginalRecordedData.append(rotetionM)
        rotetionM = np.dot(rotetionM,self.T([self.x,self.y,self.z],[0,self.yaw,round(self.pitch,3)]))
        OriginalRecordedData.append(rotetionM)
        self.rotetionL = []
        self.PoseL = []
        for rotetion in OriginalRecordedData:
            self.rotetionL.append(np.array([rotetion[0][:-1],rotetion[1][:-1],rotetion[2][:-1]]))
            self.PoseL.append(np.array([rotetion[0][-1],rotetion[1][-1],rotetion[2][-1]]))
        self.ax.quiver(self.PoseL[-1][0],self.PoseL[-1][1],self.PoseL[-1][2],0.7*self.rotetionL[-1][0][0],0.7*self.rotetionL[-1][1][0],0.7*self.rotetionL[-1][2][0],color = 'royalblue',linewidth=1.5)

    def PlotAxis(self):
        self.colorL = [['lightcoral','lightgreen','royalblue'],['lightcoral','lightgreen','royalblue'],['r','g','b']]
        self.NameL = ['map','cfm','cf']
        index = 0
        for rotetion,pose,colorm,name in zip(self.rotetionList,self.PoseList,self.colorL,self.NameL):
            self.ax.text(pose[0],pose[1],pose[2], name, color='k')
            self.ax.quiver(pose[0],pose[1],pose[2],self.arrowSizeX*rotetion[0][0],self.arrowSizeY*rotetion[1][0],self.arrowSizeZ*rotetion[2][0],color = colorm[0],linewidth=1.5)
            self.ax.quiver(pose[0],pose[1],pose[2],self.arrowSizeX*rotetion[0][1],self.arrowSizeY*rotetion[1][1],self.arrowSizeZ*rotetion[2][1],color = colorm[1],linewidth=1.5)
            self.ax.quiver(pose[0],pose[1],pose[2],self.arrowSizeX*rotetion[0][2],self.arrowSizeY*rotetion[1][2],self.arrowSizeZ*rotetion[2][2],color = colorm[2],linewidth=1.5)
            index =index +1

    def set_axes_equal(self):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def get_pointing_vector(self):
        y = np.deg2rad(self.yaw)
        p = np.deg2rad(self.pitch)
        # return np.array([np.cos(y)*np.cos(p), np.sin(y)*np.cos(p), -np.sin(p)]) # original
        self.v = np.array([np.cos(y)*np.cos(p), np.sin(p), np.sin(y)*np.cos(p)]) # Updated to match the recorded coordinated frame of the camera.

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
        self.br = tf2_ros.TransformBroadcaster()
        
        # broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        # static_transformStamped.header.stamp = rclpy.Time.now()
        static_transformStamped.header.frame_id = frame_id
        static_transformStamped.child_frame_id = child_frame_id
        # print(obj.object_name,"obj.object_name")
        static_transformStamped.transform.translation.x = float(pos[0])
        static_transformStamped.transform.translation.y = float(pos[1])
        static_transformStamped.transform.translation.z = float(pos[2])

        # quat = tf2_ros.transformations.quaternion_from_euler(            
        #         float(0),float(obj.pitch),float(obj.yaw))
        rot = Rotation.from_euler('xyz', ori, degrees=True)
            # Convert to quaternions and print
        rot_quat = rot.as_quat()
        # print(self.rot_quat)
        static_transformStamped.transform.rotation.x = rot_quat[0]
        static_transformStamped.transform.rotation.y = rot_quat[1]
        static_transformStamped.transform.rotation.z = rot_quat[2]
        static_transformStamped.transform.rotation.w = rot_quat[3]
        
        self.br.sendTransform(static_transformStamped)
    
    def T(self,Target,Orientetion):
        Orientetion = [np.deg2rad(Orientetion[0]),np.deg2rad(Orientetion[1]),np.deg2rad(Orientetion[2])]
        T = np.asarray([[1, 0 ,0,  0], [0, 1, 0, 0], [0, 0, 1, Target[2]], [0, 0, 0, 1]])
        T = np.dot(T,np.asarray([[1, 0 ,0,  Target[0]], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        T = np.dot(T, np.asarray([[1, 0 ,0, 0], [0, 1, 0, Target[1]], [0, 0, 1, 0], [0, 0, 0, 1]]))
        T = np.dot(T,np.asarray([[math.cos(Orientetion[1]), 0 ,math.sin(Orientetion[1]),0], [0,1, 0, 0], [-math.sin(Orientetion[1]), 0, math.cos(Orientetion[1]), 0], [0, 0, 0, 1]]))
        T = np.dot(T,np.asarray([[1, 0 ,0, 0], [0,math.cos(Orientetion[0]),-math.sin(Orientetion[0]), 0], [0,math.sin(Orientetion[0]),math.cos(Orientetion[0]), 0], [0, 0, 0, 1]]))       
        T = np.dot(T,np.asarray([[math.cos(Orientetion[2]), -math.sin(Orientetion[2]) ,0,0], [math.sin(Orientetion[2]),math.cos(Orientetion[2]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        return T

    def GetRobotFK(self):
        self.map = [[0,0,0],[0,0,0]]
        self.camera_frame_motive = [[0,0,1],[90,0,0]]
        self.camera_frame = [[0,0,0],[-90,0,180]]
        self.M = [self.camera_frame_motive,self.camera_frame]
        rotetionM = self.T(self.map[0],self.map[1])
        self.matrixList.append(rotetionM)
        for M in self.M:
            rotetionM = np.dot(rotetionM,self.T(M[0],M[1]))
            self.matrixList.append(rotetionM)
    
    def CreatRotetionMatrixList(self):
        self.matrixList = []
        self.GetRobotFK()
        self.rotetionList = []
        self.PoseList = []
        self.ScatterDots = [[],[],[]]
        for rotetion in self.matrixList:
            self.rotetionList.append(np.array([rotetion[0][:-1],rotetion[1][:-1],rotetion[2][:-1]]))
            self.PoseList.append(np.array([rotetion[0][-1],rotetion[1][-1],rotetion[2][-1]]))
            self.ScatterDots[0].append(rotetion[0][-1])
            self.ScatterDots[1].append(rotetion[1][-1])
            self.ScatterDots[2].append(rotetion[2][-1])
    
if __name__ == "__main__":
    rospy.init_node("test_pointer_plot")
    MathPlotLibCheck()