#!/usr/bin/python
import rospy
from std_msgs.msg import Float64MultiArray, Float64
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from unitree_legged_msgs.msg import HighCmd
import numpy as np

class Control(object):
    move2target = False
    def __init__(self):
        rospy.Subscriber('/target_new', Float64MultiArray, self.get_target)
        rospy.Subscriber('/robot_position', Float64MultiArray, self.get_position)
        rospy.Subscriber('/yaw', Float64, self.get_yaw)
        self.PubHighCmd = rospy.Publisher('high_cmd', HighCmd, queue_size=1)
        
        s = rospy.Service('trigger', Trigger, self.trigger)
        self.set_target_srv = rospy.ServiceProxy('set_moving_mode', Trigger)
        self.gps_reset_srv = rospy.ServiceProxy('gps_reset', Trigger)
        rospy.wait_for_message("/target_new", Float64MultiArray)  

        self.PubMsg = HighCmd()
        self.PubMsg.head = [254,239]
        self.PubMsg.levelFlag = 238
        self.set_default_stop()

        while not rospy.is_shutdown(): 
            self.get_angle2target() # Compute angle deviation from target
            self.set_default_stop() # Default is for the robot to stand
            print(np.rad2deg(self.angle2target), self.rotation_direction, self.get_distance2target())
            
            if self.move2target:
                if np.abs(self.angle2target) > np.deg2rad(7):
                    self.align2target()
                else:
                    self.move_forward()
                if self.get_distance2target() < 0.1:
                    self.move2target = False
                    self.set_target_srv(TriggerRequest()) # Free target
            
            self.PubHighCmd.publish(self.PubMsg)

            rospy.sleep(0.01) 

    
    def move_forward(self):
        print('Moving forward!')
        self.PubMsg.mode = 2
        self.PubMsg.gaitType = 1
        self.PubMsg.velocity[0] = 0.4
        self.PubMsg.velocity[1] = 0
        self.PubMsg.yawSpeed = 0
        self.PubMsg.footRaiseHeight = 0.1

    def align2target(self):
        print('Aligning to target!')
        self.PubMsg.mode = 2
        self.PubMsg.gaitType = 1
        self.PubMsg.velocity[0] = 0
        self.PubMsg.velocity[1] = 0
        self.PubMsg.footRaiseHeight = 0.1

        # Test these two
        self.PubMsg.yawSpeed = self.rotation_direction * 0.7
        self.PubMsg.euler[2] = self.angle2target 

    def set_default_stop(self):
        self.PubMsg.mode = 0 # See example_walk
        self.PubMsg.gaitType = 1
        self.PubMsg.speedLevel = 0
        self.PubMsg.footRaiseHeight = 0
        self.PubMsg.bodyHeight = 0
        self.PubMsg.euler[0] = 0
        self.PubMsg.euler[1] = -np.deg2rad(20)*0
        self.PubMsg.euler[2] = 0
        self.PubMsg.velocity[0] = 0.0
        self.PubMsg.velocity[1] = 0.0
        self.PubMsg.yawSpeed = 0.0
        self.PubMsg.reserve = 0

    def get_angle2target(self):
        alpha = np.arctan2(self.target[1] - self.robot_position[1], self.target[0] - self.robot_position[0])
        
        self.angle2target = alpha - self.yaw
        if self.angle2target > np.pi:
            self.angle2target -= 2*np.pi

        if np.rad2deg(np.abs(self.angle2target)) > 170:
            self.rotation_direction = -1
        else:
            self.rotation_direction = np.sign(self.angle2target)

    def get_distance2target(self):
        return np.linalg.norm([self.target[0] - self.robot_position[0], self.target[1] - self.robot_position[1]])

    def get_target(self, msg):
        self.target = msg.data

    def get_position(self, msg):
        self.robot_position = msg.data

    def get_yaw(self, msg):
        self.yaw = msg.data

    def trigger(self, request):
        self.move2target = not self.move2target

        try:
            self.set_target_srv(TriggerRequest()) # Freeze/release target
            self.gps_reset_srv(TriggerRequest())
        except:
            pass

        if self.move2target:
            print('Triggered motion to target!')
            return TriggerResponse(success=True, message="Triggered motion to target!")
        else:
            print('Stopped!')
            return TriggerResponse(success=True, message="Stopped!")
        
    
if __name__ == '__main__':
    rospy.init_node('control2target')
    Control()