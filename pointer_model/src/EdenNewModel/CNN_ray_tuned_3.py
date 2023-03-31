import numpy as np
import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from PIL import Image
import os
import warnings
import time
# import wandb

warnings.filterwarnings("ignore")


class CNN(nn.Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(12800, 64),
            nn.Sigmoid(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )  # 12800 , 4608
        self.fc2 = nn.Sequential(
            nn.Linear(12800, 8),
            nn.Sigmoid(),
            nn.Linear(8, 128),
            nn.Sigmoid()
        )
        self.lin1 = nn.Linear(4, 2)
        self.lin2 = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()
        self.to(device)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # Flattening
        x = torch.flatten(x, 1)
        output1 = self.fc1(x)
        output1 = self.lin1(output1)
        output1 = self.sigmoid(output1)
        output1 = 360 * (output1 - 0.5)  # Limit output to be at [-180, 180]
        output2 = self.fc2(x)
        output2 = self.lin2(output2)

        return output1, output2


def train(model, train_dataloader, val_dataloader, epochs, lr, device):
    since = time.time()

    params = model.parameters()

    optimizer = optim.Adam(params, lr=0.0003994042303855573, betas=[0.9286636079247964, 0.979249339931972],
                           weight_decay=0.009990026412956935)

    # scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    criterion1 = nn.MSELoss()
    criterion1 = criterion1.to(device)
    criterion2 = nn.L1Loss()
    criterion2 = criterion2.to(device)

    wandb.watch(model, log_freq=100)

    for epoch in tqdm(range(epochs)):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{epochs}')

        train_batch_loss_ang = []
        train_batch_loss_pos = []

        batch_num = 1
        for yaw_angle, pitch_angle, finger_position, img in train_dataloader:
            train_x = img.to(device)
            yaw_angle_t = yaw_angle.reshape(-1, 1)
            pitch_angle_t = pitch_angle.reshape(-1, 1)
            train_y_ang = torch.cat((pitch_angle_t, yaw_angle_t), 1).type(torch.float32)
            train_y_ang = train_y_ang.to(device)
            train_y_pos = finger_position.to(device)

            pred_ang, pred_pos = model(train_x)

            loss1 = torch.sqrt(criterion1(pred_ang, train_y_ang))
            loss2 = criterion2(pred_pos, train_y_pos)

            loss = loss1 + loss2

            wandb.log({"train_loss_angles": loss1,
                       "train_loss_position": loss2})

            train_batch_loss_ang.append(loss1.cpu().detach().numpy())
            train_batch_loss_pos.append(loss2.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            # print(f'Batch num {batch_num}/6711')
            batch_num += 1

        train_loss_av_ang = np.mean(np.array(train_batch_loss_ang))
        train_loss_av_poss = np.mean(np.array(train_batch_loss_pos))

        # validation
        val_batch_loss_ang = []
        val_batch_loss_pos = []
        with torch.no_grad():
            for yaw_angle, pitch_angle, finger_position, img in val_dataloader:
                val_x = img.to(device)
                yaw_angle_t = yaw_angle.reshape(-1, 1)
                pitch_angle_t = pitch_angle.reshape(-1, 1)
                val_y_ang = torch.cat((pitch_angle_t, yaw_angle_t), 1)
                val_y_ang = val_y_ang.to(device)
                val_y_pos = finger_position.to(device)

                val_pred_ang, val_pred_pos = model(val_x)

                loss_val_ang = torch.sqrt(criterion1(val_pred_ang, val_y_ang))
                loss_val_pos = criterion2(val_pred_pos, val_y_pos)

                # loss_val = loss_val_ang + loss_val_pos
                wandb.log({"val_loss_angles": loss_val_ang,
                           "val_position": loss_val_pos})

                val_batch_loss_ang.append(loss_val_ang.cpu().detach().numpy())
                val_batch_loss_pos.append(loss_val_pos.cpu().detach().numpy())

                del val_x, val_y_ang, val_y_pos, val_pred_ang, val_pred_pos
                torch.cuda.empty_cache()
            val_loss_av_ang = np.mean(np.array(val_batch_loss_ang))
            val_loss_av_pos = np.mean(np.array(val_batch_loss_pos))

        print(f'\nAngles Train loss: {train_loss_av_ang}, Angles Validation loss: {val_loss_av_ang}')
        print(f'\nPosition Train loss: {train_loss_av_poss}, Position Validation loss: {val_loss_av_pos}')

        # scheduler.step()
