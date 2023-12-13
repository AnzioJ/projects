import os
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random

# from ultralytics import YOLO
import cv2
import numpy as np

from torch.utils.data.dataloader import default_collate

import torch
import cv2
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

class YOLODataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None, split='train', train_split=0.7, valid_split=0.15):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform

        # 加载图像文件名
        image_filenames = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_filenames)
        n_total = len(image_filenames)
        n_train = int(n_total * train_split)
        n_valid = int(n_total * valid_split)

        if split == 'train':
            self.image_filenames = image_filenames[:n_train]
        elif split == 'valid':
            self.image_filenames = image_filenames[n_train:n_train + n_valid]
        elif split == 'test':
            self.image_filenames = image_filenames[n_train + n_valid:]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_folder, img_name)
        
        # 使用cv2.imread加载图像
        image = cv2.imread(img_path)
        
        if image is None:
            return None, None

        # 调整图像大小
        target_size = (416, 416)
        image = cv2.resize(image, target_size)

        # 转换为PIL图像进行进一步的预处理
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 应用预处理转换
        image = transform(image)

        label_path = os.path.join(self.label_folder, os.path.splitext(img_name)[0] + '.txt')
        bbox_labels = self.parse_label_file(label_path)

        return image, bbox_labels

    def parse_label_file(self, label_path):
        max_boxes = 10
        bbox_labels = torch.zeros((max_boxes, 5))
        if not os.path.exists(label_path):
            # 设置特殊标记，第一个值为1表示没有人，其余坐标为0
            bbox_labels[0] = torch.tensor([1, 0, 0, 0, 0])
            return bbox_labels

        with open(label_path, 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines[:max_boxes]):
            parts = line.strip().split()
            if len(parts) == 5:
                label, x, y, width, height = map(float, parts)
                bbox_labels[i] = torch.tensor([label, x, y, width, height])
        return bbox_labels




# 自定义 collate 函数
def custom_collate_fn(batch):
    images, bbox_labels = zip(*[(img, label) for img, label in batch if img is not None])
    images = default_collate(images)
    bbox_labels = torch.stack(bbox_labels)
    return images, bbox_labels

# 使用示例
image_folder = 'Downloads/Human_Detection/images_data_02'
label_folder = 'Downloads/Human_Detection/txt_data'
train_dataset = YOLODataset(image_folder, label_folder, split='train', train_split=0.7, valid_split=0.15)
valid_dataset = YOLODataset(image_folder, label_folder, split='valid', train_split=0.7, valid_split=0.15)
test_dataset = YOLODataset(image_folder, label_folder, split='test', train_split=0.7, valid_split=0.15)

# 使用 DataLoader 加载这些数据集
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
# 测试数据加载
for images, labels in train_dataloader:
    print(images.shape, labels.shape)
    break
for images, labels in train_dataloader:
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    # 遍历批次中的每个样本
    for i in range(labels.size(0)):  
        print(f"Sample {i}:")
        for j in range(labels.size(1)):  
            bbox = labels[i, j]
            # 检查边界框是否有效
            if torch.any(bbox != 0):
                print(f"  Box {j}: Class={bbox[0]}, X={bbox[1]}, Y={bbox[2]}, Width={bbox[3]}, Height={bbox[4]}")
            else:
                print(f"  Box {j}: Empty/Invalid box")
    break  

import torch
import torch.nn as nn

class SimpleYOLO(nn.Module):
    def __init__(self, S=7, B=1, C=1):
        super(SimpleYOLO, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 添加额外的池化层
        self.conv_out = nn.Conv2d(64, 5 * self.S * self.S, 1)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool2(self.conv3(x))  # 使用额外的池化层
        x = self.conv_out(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 5, self.S, self.S)

        return x

model = SimpleYOLO()

class CustomYOLOLoss(nn.Module):
    def __init__(self):
        super(CustomYOLOLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")  # 均方误差损失
        self.sigmoid = nn.Sigmoid()

    def forward(self, predictions, target):

        pred_cls = self.sigmoid(predictions[..., 0])  
        pred_boxes = predictions[..., 1:5]  

        target_cls = target[..., 0]
        target_boxes = target[..., 1:5]


        loss_cls = self.mse_loss(pred_cls, target_cls)
        loss_boxes = self.mse_loss(pred_boxes, target_boxes)


        total_loss = loss_cls + loss_boxes

        return total_loss


model = SimpleYOLO()
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

criterion = CustomYOLOLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

history = {
    'train_loss': [],
    'validation_loss': [],
    'train_accuracy': [],
    'validation_accuracy': []
}

best_val_loss = float('inf')
patience = 10
trigger_times = 0
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in train_dataloader:
        images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        targets = targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        optimizer.zero_grad()
        outputs = model(images)


        bbox_labels = targets[..., 0]
        labels = targets[..., 1:5].long()

        loss = criterion(outputs, bbox_labels, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_dataloader)
    train_accuracy = correct / total
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in valid_dataloader:
            images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            targets = targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            outputs = model(images)
            bbox_labels = targets[..., 1:5]
            labels = targets[..., 0].long()

            loss = criterion(outputs, bbox_labels, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_loss = running_loss / len(valid_dataloader)
    validation_accuracy = correct / total
    history['validation_loss'].append(validation_loss)
    history['validation_accuracy'].append(validation_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}')

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break

best_val_acc = max(history['validation_accuracy'])
print("Best validation accuracy is:", best_val_acc)
