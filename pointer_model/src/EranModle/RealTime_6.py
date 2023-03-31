#!/usr/bin/env /home/inbarm/catkin_ws/src/HRI_Project/HRI_Navigetion_project/pointer_model/src/melodic_py3/bin/python3
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms

# print(torch.__version__)
# print(torchvision.__version__)

from Seg_model_Lisa import FPN
import Seg_util_Lisa
from torchvision import transforms
import warnings
import time
from Model import CNN
from torchvision.models import ResNet50_Weights

warnings.filterwarnings("ignore")


# print(torch.__version__)
# print(torchvision.__version__)

def countdown(time_factor):
    time.sleep(5)
    print(" ")
    print("Starts in:")
    for i in range(0, 3):
        print(3 - i)
        time.sleep(time_factor)
    print(" ")
    time.sleep(2)


def real_time_seg(in_frame, seg_transforms, model):
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

    c_mask = np.zeros_like(in_frame)
    c_mask[:, :, 0] += (mask).astype('uint8')
    c_mask[:, :, 1] += (mask).astype('uint8')
    c_mask[:, :, 2] += (mask).astype('uint8')
    res = ((in_frame / c_mask)* c_mask).astype('uint8')
    return masked, res

def pde(in_frame, main_model, main_model_transform, pre_model, mask_model, seg_transforms, midas_transform, device):

    mask = real_time_seg(in_frame, seg_transforms, mask_model)

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

    depth_map, res = comb_mask2image(mask, depth_map)

    depth_map_T = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
    depth_map_T = Image.fromarray(depth_map_T.astype(np.uint8), mode='RGB')
    depth_map_T = main_model_transform(depth_map_T)
    depth_map_T = depth_map_T[None, :]
    # depth_map_T = depth_map_T.to(device)


    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    depth_map_T = preprocess(depth_map_T) ##########################################
    in_frame = main_model_transform(Image.fromarray(in_frame))
    in_frame = preprocess(in_frame).unsqueeze(0)
    res = preprocess(Image.fromarray(res)).unsqueeze(0)
    X_comb = torch.cat((in_frame, res, depth_map_T), dim=1).to(device)

    with torch.no_grad():
        yaw_pred, pitch_pred, pos_pred = main_model(X_comb)

    yaw_pred, pitch_pred, pos_pred = yaw_pred.cpu().numpy(), pitch_pred.cpu().numpy(), pos_pred.cpu().numpy()
    yaw = yaw_pred[0]
    pitch = pitch_pred[0]
    position = pos_pred[0]
    result = np.concatenate((yaw, pitch, position), axis=None)
    np.set_printoptions(suppress=True)
    return result, depth_map


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # # Main model:
    main_model_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # weights_path = '/home/eranbamani/Documents/weights/Estimator/net_02_04_2023/net_Sat_Feb__4_14_28_51_2023.pt'
    weights_path = '/home/roblab21/catkin_ws/src/pointer_model/src/EranModle/net_Sat_Feb__4_14_28_51_2023.pt'

    main_model = CNN()
    main_model.load_state_dict(torch.load(weights_path, map_location=device))
    main_model.eval()
    main_model = main_model.to(device)

    # # Arm mask model:
    # path_to_lisas_weights = '/home/eranbamani/Documents/PointProject/Segmentation/checkpoint/LISA/checkpoint27.pt'
    path_to_lisas_weights = '/home/roblab21/catkin_ws/src/pointer_model/src/EranModle/checkpoint27.pt'
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
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('pointing.mp4')
    start = 0
    while True:
        success, frame = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        if start >= 10:
            # print(frame.shape)
            result, md_frame = pde(frame, main_model, main_model_transform,
                                   midas_model, mask_model, seg_transforms,
                                   midas_transform, device)
            print(result)
            print(start)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(md_frame,
                        str([result[0], result[1]]),
                        (50, 50),
                        font, 0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)
            # str([result[0], result[0]])
            # frame = cv2.cvtColor(md_frame, cv2.COLOR_RGB2GRAY)
            # frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
            cv2.imshow('my model', md_frame)
        start += 1
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    # cap = cv2.imread('4.png')
    # start = time.time()
    # #
    # results, md_frame = pde(cap, main_model, pre_model, mask_model, transform, device)
    # cv2.imshow('my model', md_frame)
    # now = time.time()
    # print(results)
    # print('time:', now - start)
    # cv2.waitKey()