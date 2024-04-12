from ultralytics import YOLO
import torch
model = YOLO("yolov8n.yaml")

device = torch.device("cuda")
# 检查是否有可用的 GPU
# if torch.cuda.is_available():
#     device = torch.device("cuda")  # 如果有 GPU，则使用 GPU
#     print(device)
# else:
#     device = torch.device("cpu")  # 如果没有 GPU，则使用 CPU
#     print(device)
# model.to(device)

if __name__ == '__main__':
    results = model.train(data="config.yaml",epochs = 10)