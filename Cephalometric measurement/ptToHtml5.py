import torch
import torchvision
from module_Resnet50 import ResNet_dilated as coarse_model


# 定义模型结构
model = coarse_model()
# 加载模型参数
model.load_state_dict(torch.load('../Graduation_Project/preModel/preModel.pt'))
model.eval()
# 保存模型
torch.save(model, '../Graduation_Project/preModel.pt')
model = torch.load('../Graduation_Project/preModel.pt', map_location=torch.device('cpu'))
# 创建一个输入样本，需要和模型的输入数据形状一致
dummy_input = torch.randn(1, 1, 160, 160)
# 将模型转换成ONNX格式
torch.onnx.export(model, dummy_input, '../Graduation_Project/onnx_model.onnx', verbose=True)



