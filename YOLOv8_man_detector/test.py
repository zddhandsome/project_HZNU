from ultralytics import YOLO
import torch

model = YOLO("runs/detect/train22/weights/best.pt")

# model = torch.load("runs/detect/train22/weights/best.pt")

img  ="IMG_20240120_160005.jpg"

result = model(img,save = True)

# 打印结果或进行其他处理
# result.print()  # 打印结果到控制台
# result.show()  # 可视化结果
