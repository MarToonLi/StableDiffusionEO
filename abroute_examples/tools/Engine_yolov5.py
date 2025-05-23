from collections import OrderedDict
import numpy as np
import onnx
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.backend.trt import util as trt_util
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
import requests
from io import BytesIO
from cuda import cudart

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None



class Engine():
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.latent_h = 1120
        self.latent_w = 1120
        self.batch_size = 1
        self.cuda_graph_instance = None # cuda graph
        
    def yolov5_model_shape_dict(self):
        return {
            'images': (self.batch_size, 3, self.latent_h, self.latent_w),
            'output0': (self.batch_size, 77175, 14)
        }
    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray) ]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        """激活TensorRT引擎的执行上下文。
        """
        # 如果提供了 reuse_device_memory，则创建一个没有分配新设备内存的执行上下文，并将其设置为当前上下文
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:  # 如果没有提供 reuse_device_memory，则创建一个新的执行上下文
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        """为 TensorRT 引擎分配输入和输出缓冲区。
        """
        # 遍历每个绑定（输入或输出）
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]                 # 获取绑定的名称
            if shape_dict and binding in shape_dict:   # 如果提供了 shape_dict 并且绑定在 shape_dict 中，则使用 shape_dict 中的形状
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)         # 否则，使用引擎的默认形状
            dtype = trt.nptype(self.engine.get_binding_dtype(binding)) # 获取绑定的数据类型
            if self.engine.binding_is_input(binding):        # 如果绑定是输入，则设置上下文的绑定形状
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor                   # 将张量添加到 tensors 字典中，键为绑定的名称
            
    def get_engine_infor(self):
        # 计算输入张量的数量
        nInput = np.sum([self.engine.binding_is_input(i) for i in range(self.engine.num_bindings)])
        # 计算输出张量的数量
        nOutput = self.engine.num_bindings - nInput
        # 获取输入张量的名称和形状
        input_infor = dict((self.engine.get_tensor_name(i), self.context.get_binding_shape(i))  for i in range(nInput))
        # 获取输出张量的名称和形状
        ouput_infor = dict((self.engine.get_tensor_name(nInput + i), self.context.get_binding_shape(nInput + i))  for i in range(nOutput))
        print("TensorRT engine infors -----------------")
        print("engin nInput: ", nInput, ", Input shape: ", input_infor)
        print("engin nOutput: ", nOutput, ", Outpu shape: ", ouput_infor)

    def infer(self, feed_dict, stream=None, use_cuda_graph=False):
        # import pdb; pdb.set_trace()
        # 遍历 feed_dict 中的每个键值对，将值复制到对应的张量中
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        # 遍历 tensors 中的每个键值对，设置张量的地址
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # 如果使用 CUDA 图优化
        if use_cuda_graph:
            
            if self.cuda_graph_instance is not None:  # 如果已经有一个 CUDA 图实例
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))  # 启动 CUDA 图实例
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))                      # 同步 CUDA 流
            else:                                     # 如果没有 CUDA 图实例
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror: raise ValueError(f"ERROR: inference failed.")   # 如果推理失败，抛出异常
                # capture cuda graph
                CUASSERT(cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))                # 开始捕获 CUDA 流
                self.context.execute_async_v3(stream.ptr)    # 执行推理
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))   # 结束捕获 CUDA 流
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))  # 实例化 CUDA 图
        else:
            if stream:
                noerror = self.context.execute_async_v3(stream.ptr)
            else:
                noerror = self.context.execute_async_v3(0)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

        return self.tensors


