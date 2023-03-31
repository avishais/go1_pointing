import numpy as np
import torch
import cv2
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
import warnings
import time

warnings.filterwarnings("ignore")


class CNN(nn.Module):
    def __init__(self, device):
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
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4608, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 225),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4608, 4096),
            nn.Tanh(),
            nn.Linear(4096, 225),
            nn.Tanh()
        )
        self.lin1 = nn.Linear(225, 2)
        self.lin2 = nn.Linear(225, 3)
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
    optimizer = optim.SGD(params, lr=lr, momentum=0.9)
    criterion1 = nn.MSELoss()
    criterion1 = criterion1.to(device)
    criterion2 = nn.L1Loss()
    criterion2 = criterion2.to(device)
    # criterion2 = nn.MSELoss()
    # criterion2 = criterion2.to(device)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    losses_ang = []
    losses_pos = []
    losses_val_ang = []
    losses_val_pos = []

    for epoch in tqdm(range(epochs)):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{epochs}')

        train_batch_loss_ang = []
        train_batch_loss_pos = []

        scheduler.step()

        batch_num = 1
        for train_x, train_y in train_dataloader:
            train_x = train_x.to(device)
            train_y = train_y.to(device)

            train_y_ang = train_y[:, 0:2]
            train_y_pos = train_y[:, 2:]

            pred_ang, pred_pos = model(train_x)

            loss1 = torch.sqrt(criterion1(pred_ang, train_y_ang))
            loss2 = criterion2(pred_pos, train_y_pos)

            loss = loss1 + loss2

            train_batch_loss_ang.append(loss1.cpu().detach().numpy())
            train_batch_loss_pos.append(loss2.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_num += 1

        train_loss_av_ang = np.mean(np.array(train_batch_loss_ang))
        train_loss_av_poss = np.mean(np.array(train_batch_loss_pos))

        # validation
        val_batch_loss_ang = []
        val_batch_loss_pos = []
        with torch.no_grad():
            for val_x, val_y in val_dataloader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                val_y_ang = val_y[:, 0:2]
                val_y_pos = val_y[:, 2:]

                val_pred_ang, val_pred_pos = model(val_x)

                loss_val_ang = torch.sqrt(criterion1(val_pred_ang, val_y_ang))
                loss_val_pos = criterion2(val_pred_pos, val_y_pos)

                # loss_val = loss_val_ang + loss_val_pos
                val_batch_loss_ang.append(loss_val_ang.cpu().detach().numpy())
                val_batch_loss_pos.append(loss_val_pos.cpu().detach().numpy())

                del val_x, val_y_ang, val_y_pos, val_pred_ang, val_pred_pos
            val_loss_av_ang = np.mean(np.array(val_batch_loss_ang))
            val_loss_av_pos = np.mean(np.array(val_batch_loss_pos))

        # losses.append(train_loss_av)
        losses_ang.append(train_loss_av_ang)
        losses_pos.append(train_loss_av_poss)
        losses_val_ang.append(val_loss_av_ang)
        losses_val_pos.append(val_loss_av_pos)
        # losses_val.append(val_loss_av)

        print(f'\nAngles Train loss: {train_loss_av_ang}, Angles Validation loss: {val_loss_av_ang}')
        print(f'\nPosition Train loss: {train_loss_av_poss}, Position Validation loss: {val_loss_av_pos}')

    x_ax = np.array(range(epochs))
    y1 = losses_ang
    y2 = losses_pos
    y3 = losses_val_ang
    y4 = losses_val_pos
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Loss')
    ax1.plot(x_ax, y1, label='train')
    ax1.plot(x_ax, y3, 'tab:red', label='val')
    ax1.set_title("Angles")
    ax2.plot(x_ax, y2, label='train')
    ax2.plot(x_ax, y4, 'tab:orange', label='val')
    ax1.set_title("Position")
    for ax in fig.get_axes():
        ax.label_outer()
        ax.legend()
    fig.savefig('Losses7.png', bbox_inches="tight")

    now = time.time() - since
    print(f'run time: {now//60} minutes')


if __name__ == "__main__":
    BATCH_SIZE = 16
    EPOCHS = 16
    LR = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    labels_file = 'train_set/full_data_labels.xlsx'
    # img_dir = 'train_set/train_images'
    img_dir = 'train_set/train_set_depth'
    # test_label_file = 'test_set/target_0.xlsx'
    # test_img_dir = 'test_set/test_data_0'

    full_dataset = GetDataset(labels_file, img_dir,
                              transform=transforms.Compose([transforms.Resize((160, 160)),
                                                            transforms.ToTensor()]))

    # train_dataset = GetDataset(labels_file, img_dir,
    #                            transform=transforms.Compose([transforms.Resize((256, 256)),
    #                                                          transforms.ToTensor()]))
    # val_dataset = GetDataset(test_label_file, test_img_dir,
    #                          transform=transforms.Compose([transforms.Resize((256, 256)),
    #                                                        transforms.ToTensor()]))

    train_size = int(19725 * 0.8)
    test_size = 19725 - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True, drop_last=True)

    model = CNN(device)

    train(model, train_dataloader, val_dataloader, EPOCHS, LR, device)
    PATH = '/home/edenn/PycharmProjects/POS_and_ANG/position/POS_and_ANG_7.pth'
    torch.save(model.state_dict(), PATH)
