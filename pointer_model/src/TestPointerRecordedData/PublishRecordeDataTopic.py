#!/usr/bin/env /home/roblab21/catkin_ws/src/pointer_model/src/melodic_py3/bin/python3
import rospy
from pointer_model.msg import ModleData 
from std_srvs.srv import Empty
import geometry_msgs.msg
from std_msgs.msg import Header , ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from std_srvs.srv import Empty , EmptyResponse

class PublishREcorded(object):
    def __init__(self):
        self.publisher_ = rospy.Publisher('PubPointRecored', ModleData, queue_size=10)
        self.i = 0
        path = '/home/roblab21/catkin_ws/src/pointer_model/src/TestPointerRecordedData/data_record_Sun_Nov_27_17_25_09_2022.pkl'
        marker_data = self.pre_processing(path)
        
 
        self.yawL = np.array(marker_data['theta_yaw'])
        self.pitchL = np.array(marker_data['pitch_deg'])
        self.pos = np.array(marker_data['finger_position'])
        self.A_finger = marker_data['finger_pose']
        self.A_camera = marker_data['camera_pose']
        while not rospy.is_shutdown():
            print(self.i)
            StartAnimetion = rospy.get_param('StartAnimetion')
            if self.i <len(self.yawL) and StartAnimetion:  
                msg = ModleData()
                msg.x_cord = float(self.pos[self.i][0])
                msg.y_cord = float(self.pos[self.i][1])
                msg.z_cord = float(self.pos[self.i][2])
                msg.pitch = float(self.pitchL[self.i])
                msg.yaw = float(self.yawL[self.i])
                self.publisher_.publish(msg)
                self.i = self.i +1
            if self.i == len(self.yawL) or not StartAnimetion:
                rospy.set_param('StartAnimetion',False)
                self.i = 0
                msg = ModleData()
                msg.x_cord = float(0)
                msg.y_cord = float(0)
                msg.z_cord = float(0)
                msg.pitch = float(0)
                msg.yaw = float(0)
            self.publisher_.publish(msg)
            rospy.sleep(0.05)
            # print('Pub')
        
 
    def start_animetion_srv_func(self, request):
        rospy.set_param('StartAnimetion',True)  
        self.StartAnimetion = rospy.get_param('StartAnimetion')
       
        return EmptyResponse

    def pre_processing(self,full_file_paths):
        Dtrain = []
        cat = full_file_paths
        data = pickle.load(file=open(cat, "rb"))
        marker_data = data['marker_data']
        # for i in range(len(data['time_stamps'])):
        #     data_train = [marker_data['theta_yaw'][i], marker_data['pitch_deg'][i],
        #                             marker_data['finger_position'][i], marker_data['id_name'][i]]
        #     Dtrain.append(data_train)

        # data_array = np.array(Dtrain)
        return marker_data  
    

if __name__ == "__main__":
    rospy.init_node('PubRecordeData')
    PublishREcorded()