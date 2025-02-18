import numpy as np
import os
import tensorrt as trt

def onnx2trt(onnxFile, plan_name, min_shapes, opt_shapes, max_shapes, max_workspace_size = None, use_fp16=False, builder_opt_evel=None):
    logger = trt.Logger(trt.Logger.VERBOSE)
     # TensorRT 引擎构建器
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # 创建计算图网络
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    if max_workspace_size:
        config.max_workspace_size = max_workspace_size
    else:
        config.max_workspace_size = 10<<30
    
    # 解析ONNX模型
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        # import pdb; pdb.set_trace()
        (onnx_path, _) = os.path.split(onnxFile)
        if not parser.parse(model.read(), path=onnxFile):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
    print("Succeeded parsing ONNX file!")

    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if builder_opt_evel:
        config.builder_optimization_level = builder_opt_evel

    # set profile
    assert network.num_inputs == len(min_shapes)
    assert network.num_inputs == len(opt_shapes)
    assert network.num_inputs == len(max_shapes)

    # 创建一个优化配置文件，
    profile = builder.create_optimization_profile()
    # 为模型的输入张量定义一系列可能的形状范围，
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        profile.set_shape(input_tensor.name, tuple(min_shapes[i]), tuple(opt_shapes[i]), tuple(max_shapes[i]))
    # 将优化配置文件添加到构建器配置中
    config.add_optimization_profile(profile)
    # 根据 config 和 network 构建引擎
    engine = builder.build_engine(network, config)
    if not engine:
        raise RuntimeError("build_engine failed")
    print("Succeeded building engine!")

    print("Serializing Engine...")
    # 序列化引擎
    serialized_engine = engine.serialize()
    if serialized_engine is None:
        raise RuntimeError("serialize failed")
    # 保存序列化引擎
    (plan_path, _) = os.path.split(plan_name)
    os.makedirs(plan_path, exist_ok=True)
    with open(plan_name, "wb") as fout:
        fout.write(serialized_engine)

def export_clip_model():
    # plan_path是tensorRT 引擎文件
    onnx_path = "./onnx/CLIP.onnx"
    plan_path = "./engine/CLIP.plan"
    #   CLIP 模型的输入张量的最小、最优和最大形状范围
    onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)])

    # onnx2trt(onnx_path, plan_path, [(1, 1)], [(1, 77)], [(1, 128)], use_fp16=True)
    print("======================= CLIP onnx2trt done!")

def export_control_net_model():
    def get_shapes(B, S):
        return [(B, 4, 32, 48), (B, 3, 256, 384), tuple([B]), (B, S, 768)]

    onnx_path = "./onnx/ControlNet.onnx"
    plan_path = "./engine/ControlNet.plan"
    #   controlnet模型的输入张量的最小、最优和最大形状范围
    onnx2trt(onnx_path, plan_path,
             get_shapes(1, 1),
             get_shapes(1, 77),
             get_shapes(1, 128))

        # plan_path = plan_path_prefix + str(l) + "_fp16.plan"
        # onnx2trt(onnx_path, plan_path,
                 # get_shapes(1, 1),
                 # get_shapes(1, 77),
                 # get_shapes(1, 128),
                 # use_fp16=True)

    print("======================= ControlNet onnx2trt done!")

def export_controlled_unet_model():
    # (B, 4, 32, 48)是噪声输入形状，tuple([B]) 是步长， (B, S, 768) 是文本嵌入，其他是13个层的controlnet 的输入
    def get_shapes(B, S):
        return [(B, 4, 32, 48), tuple([B]), (B, S, 768),
                (B, 320, 32, 48),
                (B, 320, 32, 48),
                (B, 320, 32, 48),
                (B, 320, 16, 24),
                (B, 640, 16, 24),
                (B, 640, 16, 24),
                (B, 640, 8, 12),
                (B, 1280, 8, 12),
                (B, 1280, 8, 12),
                (B, 1280, 4, 6),
                (B, 1280, 4, 6),
                (B, 1280, 4, 6),
                (B, 1280, 4, 6)]
    
    onnx_path = "./onnx/ControlledUnet"
    onnx_path = onnx_path + "/ControlledUnet.onnx"

    plan_path = "./engine/ControlledUnet.plan"
    # Batchsize 都是 1，文本嵌入长度在1-128
    onnx2trt(onnx_path, plan_path,
             get_shapes(1, 1),
             get_shapes(1, 77),
             get_shapes(1, 128))

    # onnx2trt(onnx_path, plan_path,
             # get_shapes(1, 1),
             # get_shapes(1, 77),
             # get_shapes(1, 128),
             # use_fp16=True)

    print("======================= ControlNet onnx2trt done!")

def export_decoder_model():
    onnx_path = "./onnx/Decoder.onnx"
    plan_path = "./engine/Decoder.plan"
    # 潜在表示的形状为 4, 32, 48
    onnx2trt(onnx_path, plan_path,
            [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)])

    # onnx2trt(onnx_path, plan_path,
            # [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)],
             # use_fp16=True)

    print("======================= Decoder  onnx2trt done!")

def main():
    export_clip_model()
    # export_control_net_model()
    # export_controlled_unet_model()
    # export_decoder_model()

if __name__ == '__main__':
    main()
