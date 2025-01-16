import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch2trt
from torch2trt import torch2trt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os

def preprocess_image(image_path, input_shape):
    """
    对输入图像进行预处理。

    :param image_path: 输入图像的路径
    :param input_shape: 模型期望的输入形状 (batch_size, channels, height, width)
    :return: 预处理后的图像数据
    """
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 调整图像大小
    image = cv2.resize(image, (input_shape[3], input_shape[2]))

    # 转换为 PyTorch 张量
    image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()

    # 归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = transforms.Normalize(mean, std)(image)

    # 添加批次维度
    image = image.unsqueeze(0)

    return image

def postprocess_detections(detections, confidence_threshold=0.5):
    """
    对模型的检测结果进行后处理。

    :param detections: 模型的检测结果
    :param confidence_threshold: 置信度阈值
    :return: 过滤后的检测结果
    """
    # 过滤低置信度的检测结果
    filtered_detections = [d for d in detections if d['score'] > confidence_threshold]

    return filtered_detections

def convert_to_trt(model, input_shape, output_path):
    """
    将 PyTorch 模型转换为 TensorRT 模型并保存为 .trt 文件。

    :param model: PyTorch 模型
    :param input_shape: 输入数据的形状 (batch_size, channels, height, width)
    :param output_path: 输出 TensorRT 模型文件的路径
    """
    # 设置模型为评估模式
    model.eval()

    # 创建示例输入数据
    x = torch.ones(input_shape).cuda()

    # 将 PyTorch 模型转换为 TensorRT 模型
    model_trt = torch2trt(model, [x])

    # 保存 TensorRT 模型
    torch.save(model_trt.state_dict(), output_path)

def load_trt_model(model, trt_path):
    """
    加载 TensorRT 模型。

    :param model: PyTorch 模型
    :param trt_path: TensorRT 模型文件的路径
    :return: 加载了 TensorRT 模型的 PyTorch 模型
    """
    # 创建 TRTModule 实例
    model_trt = torch2trt.TRTModule()

    # 加载 TensorRT 模型的状态字典
    model_trt.load_state_dict(torch.load(trt_path))

    # 将 TRTModule 实例赋值给原始模型
    model = model_trt

    return model

def perform_inference(model, image_path, input_shape):
    """
    执行模型推理。

    :param model: 加载了 TensorRT 模型的 PyTorch 模型
    :param image_path: 输入图像的路径
    :param input_shape: 模型期望的输入形状 (batch_size, channels, height, width)
    :return: 推理结果
    """
    # 数据预处理
    input_data = preprocess_image(image_path, input_shape)

    # 模型推理
    with torch.no_grad():
        detections = model(input_data)

    # 数据后处理
    filtered_detections = postprocess_detections(detections[0])

    return filtered_detections

def main():
    # 加载预训练的缺陷检测模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.cuda()

    # 定义输入形状
    input_shape = (1, 3, 224, 224)

    # 输出 TensorRT 模型文件的路径
    output_path = 'defect_detection_model.trt'

    # 将模型转换为 TensorRT 模型并保存
    convert_to_trt(model, input_shape, output_path)

    # 加载 TensorRT 模型
    loaded_model = load_trt_model(model, output_path)

    # 执行推理
    image_path = 'path_to_your_image.jpg'
    detections = perform_inference(loaded_model, image_path, input_shape)

    print("检测结果:", detections)

if __name__ == "__main__":
    main()
