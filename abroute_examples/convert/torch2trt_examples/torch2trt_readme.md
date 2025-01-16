1. https://zhuanlan.zhihu.com/p/570822430: 
- 很难想象都2024年了还这么麻烦
- torch2trt模块应该只支持只支持pytorch动态模型，量化后的模型不能被torch2trt转换
(通常，静态模型是指在模型定义时就已经确定了输入的形状和数据类型，而动态模型则允许在运行时根据输入的实际情况来确定形状和数据类型。)

2. 通过torch2trt库转换的trt模型同样可以通过pth后缀的模型文件承载。但是两者的内容结构是不一致的。

3. torch2trt.TRTModule相当于torch.nn.Module，它们都是模型类。

4. torch2trt库对pytorch模型的转换，只需要使用torch2trt.torch2trt函数即可。

   该函数的参数：

   ```python
   def torch2trt(module,                    # torch.nn.Module
                 inputs,                    # 输入张量
                 input_names=None,          # 类似于torch2onnx需要指定输入张量名字和输出张量名字
                 output_names=None,         # 
                 log_level=trt.Logger.ERROR,
                 fp16_mode=False,           # 是否启用 FP16 精度
                 max_workspace_size=1<<25,  # TensorRT 构建器的最大工作空间大小，即32M
                 strict_type_constraints=False,   # 是否启用严格的类型约束
                 keep_network=True,         # 是否保留 TensorRT 网络
                 int8_mode=False,           # 是否启用 INT8 量化
                 int8_calib_dataset=None,   # INT8 校准数据集
                 int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM,  # INT8 校准算法
                 use_onnx=False,            # 是否使用 ONNX 格式进行转换
                 default_device_type=trt.DeviceType.GPU,  # 默认设备类型
                 dla_core=0,
                 gpu_fallback=True,  # 是否启用 GPU 回退
                 device_types={},
                 min_shapes=None,    # 输入张量的最小形状
                 max_shapes=None,    # 输入张量的最大形状
                 opt_shapes=None,    # 输入张量的优化形状
                 onnx_opset=None,    # ONNX 操作集版本
                 max_batch_size=None,   # 最大批处理大小
                 avg_timing_iterations=None,  # 平均计时迭代次数
                 **kwargs):
       pass
   
   # input_names和output_names参数：能够提升推理调试时模型内容的可读性，同时它们也是onnx模型的一部分。
   # max_workspace_size参数：6 * (1 << 30) 表示6G；
   ```

5. 为什么模型转换时需要指定模型的eval模式？

   model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().cuda()

   \# 在评估模式下，模型不会应用 Dropout 和 Batch Normalization 等技术，这可以提高模型在推理阶段的准确性。

   \# 在评估模式下，模型的行为是确定性的，这意味着对于相同的输入，模型将始终产生相同的输出。这对于模型的可重复性非常重要，特别是在需要进行模型比较或验证的情况下。（避开了Dropout 和 Batch Normalization、一开始的随机初始化和数据预处理时的数据增强）

6. torch2trt的重要模块

   `torch2trt` 是一个用于将 PyTorch 模型转换为 TensorRT 模型的工具库。它的主要目标是加速 PyTorch 模型在 NVIDIA GPU 上的推理性能。以下是 `torch2trt` 中的一些重要模块和它们的功能：

   1. **`torch2trt` 模块**：
      - **功能**：这是 `torch2trt` 的核心模块，提供了将 PyTorch 模型转换为 TensorRT 模型的主要接口。
      - 关键函数
        - `torch2trt(model, inputs, ...)`：将 PyTorch 模型转换为 TensorRT 模型。
        - `TRTModule`：一个 PyTorch 模块，用于加载和运行 TensorRT 模型。
   2. **`converters` 模块**：
      - **功能**：包含了各种 PyTorch 操作到 TensorRT 操作的转换器。
      - 关键函数
        - `get_equivalent_ops()`：返回一个字典，其中包含了 PyTorch 操作到 TensorRT 操作的映射。
        - `register_op()`：注册一个新的 PyTorch 操作到 TensorRT 操作的转换器。
   3. 📕**`plugins` 模块**：
      - **功能**：提供了对 TensorRT 插件的支持，允许用户自定义操作。
      - 关键函数
        - `register_plugin()`：注册一个新的 TensorRT 插件。
        - `get_plugin()`：获取已注册的 TensorRT 插件。
   4. **`tracer` 模块**：
      - **功能**：用于跟踪 PyTorch 模型的计算图，以便进行转换。
      - 关键函数
        - `trace(model, inputs)`：跟踪 PyTorch 模型的计算图。
        - `get_traced_graph()`：获取跟踪后的计算图。
   5. 📕**`utils` 模块**：
      - **功能**：包含了一些实用工具函数，用于处理模型转换过程中的各种任务。
      - 关键函数
        - `create_trt_engine()`：创建一个 TensorRT 引擎。
        - `save_engine()`：保存 TensorRT 引擎到文件。
        - `load_engine()`：从文件加载 TensorRT 引擎。
   6. 📕**`onnx_utils` 模块**：
      - **功能**：提供了与 ONNX 格式相关的工具函数，用于在 PyTorch 和 TensorRT 之间进行转换。
      - 关键函数
        - `convert_to_onnx(model, inputs)`：将 PyTorch 模型转换为 ONNX 格式。
        - `load_onnx_model()`：从文件加载 ONNX 模型。

   这些模块共同构成了 `torch2trt` 的功能，使得用户可以方便地将 PyTorch 模型转换为 TensorRT 模型，并在 NVIDIA GPU 上进行高效的推理。



