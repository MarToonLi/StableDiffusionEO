#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>

// 定义模型输入的尺寸
const int INPUT_C = 3;
const int INPUT_H = 224;
const int INPUT_W = 224;

// 定义预处理函数
void preprocessImage(const std::string& imagePath, float* inputData) {
    cv::Mat image = cv::imread(imagePath);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(INPUT_W, INPUT_H));

    // 归一化
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F, 1.0 / 255.0);
    cv::subtract(floatImage, cv::Scalar(0.485, 0.456, 0.406), floatImage);
    cv::divide(floatImage, cv::Scalar(0.229, 0.224, 0.225), floatImage);

    // 将图像数据复制到输入缓冲区
    std::memcpy(inputData, floatImage.data, INPUT_C * INPUT_H * INPUT_W * sizeof(float));
}

// 定义后处理函数
std::vector<float> postprocessDetections(float* outputData, int outputSize, float confidenceThreshold) {
    std::vector<float> filteredDetections;
    for (int i = 0; i < outputSize; i += 7) {
        float score = outputData[i + 2];
        if (score > confidenceThreshold) {
            for (int j = 0; j < 7; ++j) {
                filteredDetections.push_back(outputData[i + j]);
            }
        }
    }
    return filteredDetections;
}

// 定义日志记录器
class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity != nvinfer1::ILogger::Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

// 定义模型转换函数
void convertToTRT(const std::string& onnxModelPath, const std::string& trtModelPath) {
    // 创建TensorRT构建器
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    // 解析ONNX模型
    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to parse ONNX model." << std::endl;
        return;
    }

    // 创建构建配置
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30); // 设置工作空间大小为1GB

    // 构建TensorRT引擎
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "Failed to build TensorRT engine." << std::endl;
        return;
    }

    // 保存TensorRT引擎
    nvinfer1::IHostMemory* serializedModel = engine->serialize();
    std::ofstream file(trtModelPath, std::ios::out | std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing TensorRT engine." << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

    // 释放资源
    file.close();
    serializedModel->destroy();
    engine->destroy();
    config->destroy();
    network->destroy();
    builder->destroy();
    parser->destroy();
}

// 定义模型加载函数
nvinfer1::ICudaEngine* loadTRTModel(const std::string& trtModelPath) {
    std::ifstream file(trtModelPath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open file for reading TensorRT engine." << std::endl;
        return nullptr;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read TensorRT engine from file." << std::endl;
        return nullptr;
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);

    // 释放资源
    runtime->destroy();

    return engine;
}

// 定义模型推理函数
std::vector<float> performInference(nvinfer1::ICudaEngine* engine, const std::string& imagePath) {
    // 创建执行上下文
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 分配输入和输出缓冲区
    float* inputData;
    float* outputData;
    cudaMalloc(&inputData, INPUT_C * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&outputData, 1000 * 7 * sizeof(float)); // 假设输出大小为1000个检测结果，每个结果7个值

    // 预处理图像
    preprocessImage(imagePath, inputData);

    // 将输入数据复制到GPU
    cudaMemcpyAsync(inputData, inputData, INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice);

    // 执行推理
    context->executeV2(&inputData);

    // 将输出数据从GPU复制到主机
    std::vector<float> output(1000 * 7);
    cudaMemcpyAsync(output.data(), outputData, 1000 * 7 * sizeof(float), cudaMemcpyDeviceToHost);

    // 同步CUDA操作
    cudaStreamSynchronize(0);

    // 后处理检测结果
    std::vector<float> filteredDetections = postprocessDetections(output.data(), 1000 * 7, 0.5);

    // 释放资源
    cudaFree(inputData);
    cudaFree(outputData);
    context->destroy();

    return filteredDetections;
}

int main() {
    // 定义ONNX模型路径和TensorRT模型路径
    std::string onnxModelPath = "path_to_your_onnx_model.onnx";
    std::string trtModelPath = "defect_detection_model.trt";

    // 将模型转换为TensorRT格式
    convertToTRT(onnxModelPath, trtModelPath);

    // 加载TensorRT模型
    nvinfer1::ICudaEngine* engine = loadTRTModel(trtModelPath);
    if (!engine) {
        return 1;
    }

    // 执行推理
    std::string imagePath = "path_to_your_image.jpg";
    std::vector<float> detections = performInference(engine, imagePath);

    // 打印检测结果
    std::cout << "检测
