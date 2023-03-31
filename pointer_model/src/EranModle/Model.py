import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
from torchvision.models import resnet50, ResNet50_Weights

warnings.filterwarnings("ignore")


# from torchvision.models.resnet import  model_urls
# resnet50 = torch.hub.load_state_dict_from_url(model_urls['resnet50'])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(12800, 512),
            nn.Sigmoid(),
            nn.Linear(512, 128),
            nn.Sigmoid())

        self.fc11 = nn.Sequential(
            nn.Linear(12800, 512),
            nn.Sigmoid(),
            nn.Linear(512, 128),
            nn.Sigmoid(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(12800, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
        )
        self.lin1 = nn.Linear(128, 1)
        self.lin11 = nn.Linear(128, 1)
        self.lin2 = nn.Linear(512, 3)
        self.sigmoid = nn.Sigmoid()
        weights = ResNet50_Weights.DEFAULT
        pretrained_net = resnet50(weights=weights)
        # pretrained_net = resnet50(weights=None)
        self.prelayer = nn.Sequential(
            nn.Conv2d(9, 3, 3),
            nn.ReLU()
        )
        self.layer0 = nn.Sequential(pretrained_net.conv1, pretrained_net.relu)
        self.layer1 = nn.Sequential(pretrained_net.maxpool, pretrained_net.layer1)
        self.layer2 = pretrained_net.layer2
        self.layer3 = pretrained_net.layer3
        self.layer4 = pretrained_net.layer4
        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 512, 3),
            nn.ReLU()
        )

    def forward(self, x):
        # Block 1
        x = self.prelayer(x)
        fm0 = self.layer0(x)  # 256
        fm1 = self.layer1(fm0)  # 128
        fm2 = self.layer2(fm1)  # 64
        fm3 = self.layer3(fm2)  # 32
        fm4 = self.layer4(fm3)  # 16
        fm5 = self.layer5(fm4)
        # Flattening
        x = torch.flatten(fm5, 1)
        yaw = self.fc1(x)
        yaw = self.lin1(yaw)
        yaw = self.sigmoid(yaw)
        yaw = 360 * (yaw - 0.5)  # Limit output to be at [-180, 180]

        pitch = self.fc1(x)
        pitch = self.lin11(pitch)
        pitch = self.sigmoid(pitch)
        pitch = 360 * (pitch - 0.5)  # Limit output to be at [-180, 180]

        position = self.fc2(x)
        position = self.lin2(position)

        return yaw, pitch, position


def train(model, train_dataloader, val_dataloader, epochs, lr, device):
    params = model.parameters()

    optimizer = optim.Adam(params, lr=0.0001, betas=[0.9473974058623147, 0.9622471564606436],
                           weight_decay=0.0087)

    criterion1 = nn.MSELoss()
    criterion1 = criterion1.to(device)
    criterion2 = nn.L1Loss()
    criterion2 = criterion2.to(device)

    for epoch in range(epochs):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{epochs}')

        train_batch_loss_yaw = []
        train_batch_loss_pitch = []
        train_batch_loss_pos = []

        batch_num = 1
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for yaw_angle, pitch_angle, finger_position, img, res, comb in pbar:
            X_comb = torch.cat((img, res, comb), dim=1).to(device)
            yaw_angle_t = yaw_angle.reshape(-1, 1).type(torch.float32).to(device)
            pitch_angle_t = pitch_angle.reshape(-1, 1).type(torch.float32)  .to(device)
            train_y_pos = finger_position.to(device)

            yaw, pitch, position = model(X_comb)

            loss1 = torch.sqrt(criterion1(yaw, yaw_angle_t))
            loss11 = torch.sqrt(criterion1(pitch, pitch_angle_t))
            loss2 = criterion2(position, train_y_pos)

            loss = loss1 + loss11 + loss2


            train_batch_loss_yaw.append(loss1.cpu().detach().numpy())
            train_batch_loss_pitch.append(loss11.cpu().detach().numpy())
            train_batch_loss_pos.append(loss2.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            batch_num += 1
            pbar.set_postfix({'Epoch': epoch,
                              'Training yaw Loss': np.mean(train_batch_loss_yaw),
                              'Training pitch Loss': np.mean(train_batch_loss_pitch),
                              'Training position Loss': np.mean(train_batch_loss_pos)
                              })

        train_loss_av_yaw = np.mean(np.array(train_batch_loss_yaw))
        train_loss_av_pitch = np.mean(np.array(train_batch_loss_pitch))
        train_loss_av_poss = np.mean(np.array(train_batch_loss_pos))

        # validation
        val_batch_loss_yaw = []
        val_batch_loss_pitch = []
        val_batch_loss_pos = []
        with torch.no_grad():
            for yaw_angle, pitch_angle, finger_position, img, res, comb in val_dataloader:
                val_x = torch.cat((img, res, comb), dim=1).to(device)
                yaw_angle_t = yaw_angle.reshape(-1, 1).type(torch.float32).to(device)
                pitch_angle_t = pitch_angle.reshape(-1, 1).type(torch.float32).to(device)
                val_y_pos = finger_position.to(device)

                yaw_val, pitch_val, position_val = model(val_x)

                loss_val_yaw = torch.sqrt(criterion1(yaw_val, yaw_angle_t))
                loss_val_pitch= torch.sqrt(criterion1(pitch_val, pitch_angle_t))
                loss_val_pos = criterion2(position_val, val_y_pos)

                val_batch_loss_yaw.append(loss_val_yaw.cpu().detach().numpy())
                val_batch_loss_pitch.append(loss_val_pitch.cpu().detach().numpy())
                val_batch_loss_pos.append(loss_val_pos.cpu().detach().numpy())

                del val_x, yaw_angle_t, pitch_angle_t, val_y_pos, yaw_val, pitch_val, position_val
                torch.cuda.empty_cache()

            val_loss_av_yaw = np.mean(np.array(val_batch_loss_yaw))
            val_loss_av_pitch = np.mean(np.array(val_batch_loss_pitch))
            val_loss_av_pos = np.mean(np.array(val_batch_loss_pos))

        print(f'\nYaw Train loss: {train_loss_av_yaw}, Yaw Validation loss: {val_loss_av_yaw}')
        print(f'\nPitch Train loss: {train_loss_av_pitch}, Pitch Validation loss: {val_loss_av_pitch}')
        print(f'\nPosition Train loss: {train_loss_av_poss}, Position Validation loss: {val_loss_av_pos}')

