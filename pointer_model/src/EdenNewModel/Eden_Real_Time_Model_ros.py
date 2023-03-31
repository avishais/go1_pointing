#!/usr/bin/env /home/roblab21/catkin_ws/src/pointer_model/src/melodic_py3/bin/python3

import numpy as np
import torch
import cv2
from PIL import Image
from Seg_model_Lisa import FPN
import Seg_util_Lisa
from torchvision import models, transforms
from CNN_ray_tuned_3 import CNN
import warnings
import time

import rospy
from std_msgs.msg import String
import sensor_msgs.msg 
# from custom_interfaces.msg import ModleData
from hri_navigation.msg import ModleData
# from cv_bridge import CvBridge
import sys
sys.path.insert(1, '/home/inbarm/catkin_ws/src/OfficialGitPkg/ros_numpy/')
# print(np.version.version)
import ros_numpy

warnings.filterwarnings("ignore")


def countdown(time_factor):
    time.sleep(5)
    print(" ")
    print("Starts in:")
    for i in range(0, 3):
        print(3 - i)
        time.sleep(time_factor)
    print(" ")
    time.sleep(2)


def real_time_seg(in_frame, seg_transforms, model,device):
    I = seg_transforms(Image.fromarray(in_frame)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(I)
    mask = output.squeeze(0)
    mask = mask > 0.5
    mask = mask.detach().cpu().numpy()
    mask = (mask * 255).astype(np.uint8).squeeze(0)
    mask = mask.astype('float32')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel)
    return mask


def comb_mask2image(mask, in_frame):
    in_frame = cv2.resize(in_frame, (288, 384))
    color_mask = np.zeros_like(in_frame)
    color_mask[:, :, 1] += (mask).astype('uint8')
    masked = cv2.addWeighted(in_frame, 0.5, color_mask, 1.0, 0.0)

    return masked


def pde(in_frame, main_model, main_model_transform, pre_model, mask_model, seg_transforms, midas_transform, device):

    mask = real_time_seg(in_frame, seg_transforms, mask_model,device)

    t_frame = midas_transform(in_frame).to(device)
    with torch.no_grad():
        prediction = pre_model(t_frame)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=in_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    depth_map = comb_mask2image(mask, depth_map)

    depth_map_T = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
    depth_map_T = Image.fromarray(depth_map_T.astype(np.uint8), mode='RGB')
    depth_map_T = main_model_transform(depth_map_T)
    depth_map_T = depth_map_T[None, :]
    depth_map_T = depth_map_T.to(device)

    with torch.no_grad():
        ang_pred, pos_pred = main_model(depth_map_T)

    ang_pred, pos_pred = ang_pred.cpu().numpy(), pos_pred.cpu().numpy()
    ang = ang_pred[0]
    xyz = pos_pred[0]
    result = np.concatenate((ang, xyz), axis=None)
    np.set_printoptions(suppress=True)
    return result, depth_map


class RunModle(object):
    def __init__(self):
        rospy.init_node('PointerDirectionNode', anonymous=True)
        publisher_ = rospy.Publisher('PubModelData', ModleData, queue_size=10)
        subscribe_ = rospy.Subscriber('/camera/color/image_raw', sensor_msgs.msg.Image, self.get_robot_cam)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # # Main model:
        main_model_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        weights_path = 'PDE_arm_02_06_2023_19_39.pt'
        main_model = CNN(device)
        main_model.load_state_dict(torch.load(weights_path, map_location=device))
        main_model.eval()

        # # Arm mask model:
        path_to_lisas_weights = 'checkpoint27.pt'
        ENCODER = "densenet201"
        ENCODER_WEIGHTS = None
        CLASSES = ['arm']
        ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
        in_channels = 3

        seg_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((384, 288))])

        mask_model = FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            in_channels=in_channels
        )

        ENCODER_WEIGHTS = Seg_util_Lisa.get_state_dict(path_to_lisas_weights)
        mask_model.load_state_dict(ENCODER_WEIGHTS)
        mask_model = mask_model.to(device)
        mask_model.eval()

        # # Pre-processing (MiDAS):
        midas_model_type = 'DPT_Hybrid'
        midas_model = torch.hub.load("intel-isl/MiDaS", midas_model_type)
        midas_model.to(device)
        midas_model.eval()

        # # Pre-processing transform:
        midas_model_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if midas_model_type == "DPT_Large" or midas_model_type == "DPT_Hybrid":
            midas_transform = midas_model_transforms.dpt_transform
        else:
            midas_transform = midas_model_transforms.small_transform

        # Uncomment to open camera or input a video:
        # cap = cv2.VideoCapture(0)
        start = 0
        rospy.wait_for_message('/camera/color/image_raw', sensor_msgs.msg.Image)
        print(self.cv_image.shape)
        while True:
            # success, frame = cap.read()

            # if not success:
            #     print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                # continue
            if start >= 10:
                result, md_frame = pde(self.cv_image, main_model, main_model_transform,
                                    midas_model, mask_model, seg_transforms,
                                    midas_transform, device)
                print(result)
                msg = ModleData()
                # define the linear x-axis velocity of /cmd_vel Topic parameter to 0.5
                msg.x_cord = float(result[2])
                msg.y_cord = float(result[3])
                msg.z_cord = float(result[4])
                msg.pitch = float(result[0])
                msg.yaw = float(result[1])
                publisher_.publish(msg)
                print(start)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(md_frame,
                            str([result[0], result[1]]),
                            (20, 50),
                            font, 0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA)
                md_frame = cv2.resize(md_frame, (640,480))
                cv2.imshow('my model', md_frame)
            start += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        # cap.release()
        cv2.destroyAllWindows()
    def get_robot_cam(self,msg):
        self.frame = msg.data
        self.cv_image = ros_numpy.numpify(msg)

if __name__ == "__main__":
    RunModle()