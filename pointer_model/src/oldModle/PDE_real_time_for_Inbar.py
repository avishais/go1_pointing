import numpy as np
import torch
import cv2
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split
import torchvision
from torchvision import models, transforms
from torch.nn import functional as func
import matplotlib.pyplot as plt
from tqdm import tqdm
# from data_loading2 import GetDataset
import pandas as pd
from PDE_POS_CNN import CNN
import warnings
import os
import time

warnings.filterwarnings("ignore")


def pde(in_frame, main_model, pre_model, transform, device):
    t_frame = transform(in_frame).to(device)
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

    transform2 = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    depth_map = Image.fromarray(depth_map.astype(np.uint8), mode='RGB')
    depth_map = transform2(depth_map)
    depth_map = depth_map[None, :]
    depth_map = depth_map.to(device)

    with torch.no_grad():
        ang_pred, pos_pred = main_model(depth_map)

    ang_pred, pos_pred = ang_pred.cpu().numpy(), pos_pred.cpu().numpy()
    ang = ang_pred[0]
    xyz = pos_pred[0]
    result = np.concatenate((ang, xyz), axis=None)
    np.set_printoptions(suppress=True)
    return result


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # # Main model:
    model_path = '/home/inbarm/catkin_ws/src/HRI_Project/hri_navigation/src/PointingModel/POS_and_ANG_6.pth'
    main_model = CNN(device)
    main_model.load_state_dict(torch.load(model_path, map_location=device))
    main_model.eval()

    # # Pre-model (MiDAS):
    pre_model_type = 'DPT_Hybrid'
    pre_model = torch.hub.load("intel-isl/MiDaS", pre_model_type)
    pre_model.to(device)
    pre_model.eval()

    # # Pre-model transform:
    pre_model_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if pre_model_type == "DPT_Large" or pre_model_type == "DPT_Hybrid":
        transform = pre_model_transforms.dpt_transform
    else:
        transform = pre_model_transforms.small_transform

    # Open camera or input a video:
    cap = cv2.VideoCapture(2)
    # cap = cv2.VideoCapture('pointing.mp4')
    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        result = pde(frame, main_model, pre_model, transform, device)
        print(result)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
        cv2.imshow('my model', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    # # See result of a various images:
    # # images_dir = 'new_framess/new_framess'
    # # images_dir = 'new_frames3/new_frames3'
    # images_dir = 'no_depth_real'
    # # images_dir = 'one'
    #
    # ang_results_l = []
    # pos_results_l = []
    # distance = []
    # start = time.time()
    # for i in tqdm(range(len(os.listdir(images_dir)))):
    #     filename = os.path.join(images_dir, f'{i}.png')
    #     cap = cv2.imread(filename)
    #     frame = cv2.cvtColor(cap, cv2.COLOR_RGB2GRAY)
    #     ang_results, pos_results = pde(cap, main_model, pre_model, transform, device)
    #     ang_results_l.append(ang_results[0])
    #     pos_results_l.append(pos_results[0])
    #     distance.append(np.linalg.norm(pos_results[0]))
    #     print(f'Angles: {ang_results[0]}')
    #     print(f'Position: {pos_results[0]}')
    #
    # now = time.time()
    # print('time:', now - start)
    # ang_results_df = pd.DataFrame(ang_results_l, columns=['pitch', 'yaw'])
    # pos_results_df = pd.DataFrame(pos_results_l, columns=['x', 'y', 'z'])
    # distance_df = pd.DataFrame(distance, columns=['distance[m]'])
    # results_df = pd.concat([ang_results_df, pos_results_df, distance_df], axis=1)
    # results_df.to_excel('DPE3_real.xlsx')
    # print(results)

    # # See result on one image only:
    # cap = cv2.imread('20.png')
    # start = time.time()
    # #
    # # frame = cv2.cvtColor(cap, cv2.COLOR_RGB2GRAY)
    # # ang_results, pos_results = pde(cap, main_model, pre_model, transform, device)
    # result = pde(cap, main_model, pre_model, transform, device)
    # now = time.time()
    # print(result)
    # print('time:', now - start)
