import numpy as np
import os
import tensorrt as trt

def onnx2trt(onnxFile, plan_name, min_shapes, opt_shapes, max_shapes, max_workspace_size = None, use_fp16=False, builder_opt_evel=None):
    logger = trt.Logger(trt.Logger.VERBOSE)

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    if max_workspace_size:
        config.max_workspace_size = max_workspace_size
    else:
        config.max_workspace_size = 5<<30

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

    # 创建优化配置文件，为每个输入张量设置形状范围；
    # 以便在运行时可以根据实际输入形状实现动态调整计算资源的分配。
    profile = builder.create_optimization_profile()  
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)  #  获取第 i 个输入张量
        profile.set_shape(input_tensor.name, tuple(min_shapes[i]), tuple(opt_shapes[i]), tuple(max_shapes[i]))
        #! 如果希望cudagrapgh支持变长输入，则需要修改onnx2trt函数的参数，增加多项profile
    # 将创建好的优化配置文件添加到配置对象 config 中
    config.add_optimization_profile(profile)  

    engine = builder.build_engine(network, config)
    if not engine:
        raise RuntimeError("build_engine failed")
    print("Succeeded building engine!")

    print("Serializing Engine...")
    serialized_engine = engine.serialize()
    if serialized_engine is None:
        raise RuntimeError("serialize failed")

    (plan_path, _) = os.path.split(plan_name)
    os.makedirs(plan_path, exist_ok=True)
    with open(plan_name, "wb") as fout:
        fout.write(serialized_engine)

def export_clip_model():
    onnx_path = "./onnx/CLIP_work1_float32_opt.onnx"
    plan_path = "./engine/CLIP_work1_float32_opt.plan"

    # base: fp16
    onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)], use_fp16=False)
    
    # plan_path = "./engine/CLIP_level0.plan"
    # onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)], use_fp16=True, builder_opt_evel=0)
    
    # plan_path = "./engine/CLIP_level1.plan"
    # onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)], use_fp16=True, builder_opt_evel=1)
    
    # plan_path = "./engine/CLIP_level2.plan"
    # onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)], use_fp16=True, builder_opt_evel=2)
    
    # plan_path = "./engine/CLIP_level3.plan"
    # onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)], use_fp16=True, builder_opt_evel=3)
    
    # plan_path = "./engine/CLIP_level4.plan"
    # onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)], use_fp16=True, builder_opt_evel=4)
    
    # plan_path = "./engine/CLIP_level5.plan"
    # onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)], use_fp16=True, builder_opt_evel=5)

    print("======================= CLIP onnx2trt done!")

def export_control_net_model():
    def get_shapes(B, S):
        return [(B, 4, 32, 48), 
                (B, 3, 256, 384), 
                tuple([B]), 
                (B, S, 768)]

    onnx_path = "./onnx/ControlNet_work3.onnx"
    plan_path = "./engine/ControlNet_work3.plan"

    # base: fp16
    onnx2trt(onnx_path, plan_path,get_shapes(1, 77),get_shapes(1, 77),get_shapes(1, 77),use_fp16=True)

    # build_level:
    # plan_path = "./engine/ControlNet_level0.plan"
    # onnx2trt(onnx_path, plan_path,
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         use_fp16=True,
    #         builder_opt_evel=0)
    
    # plan_path = "./engine/ControlNet_level1.plan"
    # onnx2trt(onnx_path, plan_path,
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         use_fp16=True,
    #         builder_opt_evel=1)
    
    # plan_path = "./engine/ControlNet_level2.plan"
    # onnx2trt(onnx_path, plan_path,
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         use_fp16=True,
    #         builder_opt_evel=2)
    
    # plan_path = "./engine/ControlNet_level3.plan"
    # onnx2trt(onnx_path, plan_path,
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         use_fp16=True,
    #         builder_opt_evel=3)
    
    # plan_path = "./engine/ControlNet_level4.plan"
    # onnx2trt(onnx_path, plan_path,
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         use_fp16=True,
    #         builder_opt_evel=4)
    
    # plan_path = "./engine/ControlNet_level5.plan"
    # onnx2trt(onnx_path, plan_path,
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         get_shapes(1, 77),
    #         use_fp16=True,
    #         builder_opt_evel=5)
    
    print("======================= ControlNet onnx2trt done!")

def export_controlled_unet_model():
    def get_shapes(B, S):
        return [(B, 4, 32, 48), 
                tuple([B]), 
                (B, S, 768),
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

    onnx_path = "./onnx/ControlledUnet_work2"
    onnx_path = onnx_path + "/ControlledUnet_work2.onnx"

    plan_path = "./engine/ControlledUnet_base.plan"

    # base: fp16
    # onnx2trt(onnx_path, plan_path, get_shapes(1, 77), get_shapes(1, 77), get_shapes(1, 77), use_fp16=True)
    
    # plan_path = "./engine/ControlledUnet_level0.plan"
    # onnx2trt(onnx_path, plan_path, get_shapes(1, 77), get_shapes(1, 77), get_shapes(1, 77), use_fp16=True, builder_opt_evel=0)
    
    # plan_path = "./engine/ControlledUnet_level1.plan"
    # onnx2trt(onnx_path, plan_path, get_shapes(1, 77), get_shapes(1, 77), get_shapes(1, 77), use_fp16=True, builder_opt_evel=1)
    
    # plan_path = "./engine/ControlledUnet_level2.plan"
    # onnx2trt(onnx_path, plan_path, get_shapes(1, 77), get_shapes(1, 77), get_shapes(1, 77), use_fp16=True, builder_opt_evel=2)
    
    # plan_path = "./engine/ControlledUnet_level3.plan"
    # onnx2trt(onnx_path, plan_path, get_shapes(1, 77), get_shapes(1, 77), get_shapes(1, 77), use_fp16=True, builder_opt_evel=3)
    
    plan_path = "./engine/ControlledUnet_level4.plan"
    onnx2trt(onnx_path, plan_path, get_shapes(1, 77), get_shapes(1, 77), get_shapes(1, 77), use_fp16=True, builder_opt_evel=4)
    
    # plan_path = "./engine/ControlledUnet_level5.plan"
    # onnx2trt(onnx_path, plan_path, get_shapes(1, 77), get_shapes(1, 77), get_shapes(1, 77), use_fp16=True, builder_opt_evel=5)

    print("======================= ControlNet onnx2trt done!")

def export_decoder_model():
    onnx_path = "./onnx/Decoder_work2.onnx"
    plan_path = "./engine/Decoder_base.plan"

    # base: fp16
    # onnx2trt(onnx_path, plan_path, [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)], use_fp16=True)
    
    plan_path = "./engine/Decoder_level0.plan"
    onnx2trt(onnx_path, plan_path, [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)], use_fp16=True, builder_opt_evel=0)

    plan_path = "./engine/Decoder_level1.plan"
    onnx2trt(onnx_path, plan_path, [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)], use_fp16=True, builder_opt_evel=1)
    
    plan_path = "./engine/Decoder_level2.plan"
    onnx2trt(onnx_path, plan_path, [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)], use_fp16=True, builder_opt_evel=2)
    
    plan_path = "./engine/Decoder_level3.plan"
    onnx2trt(onnx_path, plan_path, [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)], use_fp16=True, builder_opt_evel=3)
    
    plan_path = "./engine/Decoder_level4.plan"
    onnx2trt(onnx_path, plan_path, [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)], use_fp16=True, builder_opt_evel=4)
    
    plan_path = "./engine/Decoder_level5.plan"
    onnx2trt(onnx_path, plan_path, [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)], use_fp16=True, builder_opt_evel=5)

    print("======================= Decoder  onnx2trt done!")

def main():
    # export_clip_model()
    export_control_net_model()
    # export_controlled_unet_model()
    # export_decoder_model()

if __name__ == '__main__':
    main()
