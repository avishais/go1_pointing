# unitree_control_hri
<!-- download [unitree_ros_to_real](https://github.com/unitreerobotics/unitree_ros_to_real) and install and [unitree_legged_sdk](https://github.com/unitreerobotics/unitree_legged_sdk).\ -->
In unitree_legged_real change the ip addeess acording to the connection type:\
192.168.123.161 via Eternet (Dont Forget to change your computer static ip)\
192.168.12.1 via wifi. 
change the ip in: '/home/roblab21/catkin_ws/src/unitree_ros_to_real/unitree_legged_real/src/exe/ros_udp.cpp' file line 29

If you have problem connect to the robot follow this video: [Controlling Unitree Robots with Unitree_legged_sdk](https://www.youtube.com/watch?v=tTCbdul7xsc&t=180s).
All the documentetion the compeny send is here: https://onedrive.live.com/?authkey=%21ANY%5FFdF7KBZkuJo&id=4DE2A9A50A0B4FB2%2150156&cid=4DE2A9A50A0B4FB2

# Run Keyborad control program
roslaunch unitree_legged_real real.launch ctrl_level:=highlevel

rosrun unitree_control_hri unitree_keyboard_command.py 

rosrun unitree_control_hri publish_keyboard_command.py 

rosrun unitree_control_hri convert_cmd_vel_to_high_cmd_ros.py 

# Connect to Velodine VLP-16 lidar
[vidoe to connect to the lidar program](https://www.youtube.com/watch?v=0hrY0eFTT5g&t=285s)\
password:123\
ssh pi@192.168.12.1 
ssh unitree@192.168.123.15

<!-- get wlan0 ip
00000000 -->
# Connect to unitree cammeras
To connecte the camera you need to update the SDK on the robot computer and follow the youtube tuturial

[tuturial](https://www.youtube.com/watch?v=nafv21HeeEM) 

<!--  Notes:
password:123\
ssh unitree@192.168.123.13
cd UnitreecameraSDK-main
vi trans_rect_config.yaml
chang the UDP address on my computer using ifconfig
to save and exit file :wq

cd build
rm -rf *
make

cd ..
./bins/examples_putImages

on my cpmuter:
cd UnitreecammeraSDK
cd build
cmake ..
make
cd ..
./bins/example_getimagetrans -->


## Install Wifi 5G adapter
[follow this instraction while connecting to the internet with difftent adapter](https://gist.github.com/primaryobjects/f723b966d5f42094619f9c1048c7838b)