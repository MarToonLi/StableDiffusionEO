import cv2
import torch
import torchvision.transforms as transforms


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