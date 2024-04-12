import os

import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ONNX_EXPORT_ONLY"] = '1'
import csv
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split


from utils.func import *
from utils.dataset import CoarseDataset
from model import ResNet_dilated as coarse_model

import torch.nn.functional as F

#训练过程可视化
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
logWriter = SummaryWriter(log_dir="../Graduation_Project/data/log")

# Dataset
train_val_dataset = CoarseDataset("../Graduation_Project/trainSet/cephalometric-landmarks/cepha400/trainDataset/", flag=False)  #train_img_path
test_dataset = CoarseDataset("../Graduation_Project/trainSet/cephalometric-landmarks/cepha400/testDataset/", flag=False)

len_val_set = int(0.1 * len(train_val_dataset))
len_train_set = len(train_val_dataset) - len_val_set
# print(len(test_dataset))

batch_size = 8
workers = 0
learning_rate = 0.001
epochs = 100
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = "../Graduation_Project/weight/"
T_max = 50
coarse_model_path = "../Graduation_Project_Early_Model/earlyPreModel.pt"

train_dataset, val_dataset = random_split(train_val_dataset, [len_train_set, len_val_set],
                                          generator=torch.Generator().manual_seed(0))

print(f"train : val : test = {len(train_dataset)} : {len(val_dataset)} : {len(test_dataset)}")

# Dataloader
train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                         pin_memory=True, drop_last=True)
val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                       pin_memory=True, drop_last=True)
test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                        pin_memory=True, drop_last=True)

# Make dir
if os.path.exists(weight_path):
    shutil.rmtree(weight_path)
os.makedirs(weight_path)
f = open(f"{weight_path}log.csv", 'w', newline='')
wr = csv.writer(f)
wr.writerow(['epoch', 'train loss', 'val loss', 'LR'])
f.close()

''' Training Coarse Model '''
# Make network
# print(torch.__version__)
# print(torchvision.__version__)
# model = torchvision.models.resnet50(pretrained=False)
model = coarse_model()
model = model.cuda()
model.to(device)

print("Create coarse model")

criterion = nn.MSELoss().to(device)

# output_params = list(map(id, model.fc.parameters()))
# feature_params = filter(lambda p: id(p) not in output_params, model.parameters())
# optimizer = optim.SGD([{'params': feature_params},
#                            {'params': model.fc.parameters(), 'lr': learning_rate }],
#                           lr=learning_rate, weight_decay=0.01)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=5, after_scheduler=cosine_scheduler)

early_stopping = EarlyStopping(patience=20, path=coarse_model_path, verbose=True)



# Training
print("Start Training...")
save_point = 0
min_loss = float('inf')

global_step = 0
for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0

    model.train().to(device)
    csvfile = open('coordinates.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile)
    for batch_index, (features,labels,org) in enumerate(train_data):
        # print('Batch {}:  Features - {}, Labels - {}'.format(batch_index, features, labels))
        img = features.to(device)

        labels = torch.stack(labels).view(batch_size, -1)
        labels = np.array(labels, dtype=np.float32)
        labels = torch.as_tensor(labels).to(device)

        optimizer.zero_grad()
        
        img = img.type(torch.cuda.FloatTensor)
        
        outputs = model(img)
        

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += (loss.item() / batch_size)
        # 写入日志文件
        logWriter.add_scalar('Train/Loss', loss.item(), global_step=global_step)
        # 添加图片
        if global_step % 2 == 0:
            img_grid = torchvision.utils.make_grid(img)
            logWriter.add_image('input_images', img_grid, global_step=global_step)
        global_step += 1


        # 显示img
        for label in labels:
            plt.plot(label[0].cpu(), label[1].cpu(), 'ro')
            # 将坐标信息写入csv文件中
            csvwriter.writerow([label[0], label[1]])


    mean_train_loss = train_loss / len(train_data)

    model.eval().to(device)
    with torch.no_grad():
        for batch_index, (features,labels,org) in enumerate(val_data):
            img = features.to(device)

            labels = torch.stack(labels).view(batch_size, -1)
            labels = np.array(labels, dtype=np.float32)
            labels = torch.as_tensor(labels).to(device)

            # img = np.reshape(img, (16, 1, 224, 224)).to(device)
            print("eval.....")
            img = img.type(torch.cuda.FloatTensor)
            outputs = model(img)

            loss = criterion(outputs, labels)

            val_loss += (loss.item() / batch_size)

    mean_val_loss = val_loss / len(val_data)

    lr_scheduler.step()
    early_stopping(mean_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    # print log
    print(f"epoch: {epoch + 1} | training loss: {mean_train_loss:.8f} | validation loss: {mean_val_loss:.8f} | lr: {optimizer.param_groups[0]['lr']:.10f}")

    # Save log
    f = open(f"{weight_path}coarse_log.csv", 'a', newline='')
    wr = csv.writer(f)
    wr.writerow([epoch + 1, train_loss / len(train_data), mean_val_loss, optimizer.param_groups[0]['lr']])
    f.close()

    if mean_val_loss < min_loss:
        min_loss = mean_val_loss
        save_point = epoch + 1
        torch.save(model.state_dict(), coarse_model_path)
# 启动TensorBoard
print(f"Min val loss: {min_loss} | Save idx: {save_point}")
torch.save(model.state_dict(),"preModel.pt")
print("End Training")