
# Prerequisites

1. ROS noetic
2. pytorch

# Run model
To run the main model, run:
'''
roslaunch pointer_model test_pointer_modle.launch pointer:=modle
'''

In a diffrent terminal:
'''
rosrun pointer_model RealTime_6_pub_ros.py
'''


1. file RealTime_6_pub_ros.py read image from web cam.
2. file RealTime_6_pub_ros_from_realseance.py read ros msg Image and convert it to the network configuretion. 
to yous this you need to change the topoc name to get the camera msg (line 126).

TFs:
the camera frame is publish according to 'camera_frame' (SubscribAndPlotModleArrowsTF.py line 44) if you want to change it to your camera frame in the URDF change also int this file and the frame and the pointer will move according to the robot movment

python env:
the python env that run the net is melodic_py3.
if you need more packges install make sure you acivate the env before installetion.
cd /home/roblab21/catkin_ws/src
source melodic_py3/bin/activate

