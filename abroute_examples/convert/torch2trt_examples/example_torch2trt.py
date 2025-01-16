"""
将yolov5模型转换为tensorrt模型

torch2trt  https://github.com/NVIDIA-AI-IOT/torch2trt
最近一次的更新是在9个月之前

"""

import torch
import torchvision
from torch2trt import torch2trt

"""
model = torch.load(model_path).eval().cuda()
# 在评估模式下，模型不会应用 Dropout 和 Batch Normalization 等技术，这可以提高模型在推理阶段的准确性。
# 在评估模式下，模型的行为是确定性的，这意味着对于相同的输入，模型将始终产生相同的输出。这对于模型的可重复性非常重要，特别是在需要进行模型比较或验证的情况下。（避开了Dropout 和 Batch Normalization、一开始的随机初始化和数据预处理时的数据增强）
"""



def convert_torch_trt(model_torch, model_input):
    # 将 PyTorch 模型转换为 TensorRT 模型
    model_trt = torch2trt(model_torch, [model_input], keep_network=False)

    # 进行推理
    output = model_trt(model_input)

    # 输出结果
    print(output)
    
    return model_trt



if __name__ == '__main__':
    # 加载 PyTorch 模型
    model_torch_path = 'path/to/your/model.pth'  # 替换为你的模型文件路径
    model_trt_path = 'path/to/your/model.pth'  # 替换为你的模型文件路径
    model_torch = torch.load(model_torch_path).eval().cuda()
    
    # 转换模型
    input = torch.ones((1, 3, 224, 224)).cuda()
    model_trt = convert_torch_trt(model_torch, input)
    
    
    # 保存 TensorRT 模型
    torch.save(model_trt.state_dict(), model_trt_path)