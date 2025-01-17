import numpy as np
import os
import tensorrt as trt

def onnx2trt(onnxFile, plan_name, min_shapes, opt_shapes, max_shapes, max_workspace_size = None, use_fp16=False, builder_opt_evel=None):
    logger = trt.Logger(trt.Logger.VERBOSE)                                                         # create logger
    builder = trt.Builder(logger)                                                                   # create builder
    config = builder.create_builder_config()                                                        # create config
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # create network
    
    if max_workspace_size:                                                                          # init config
        config.max_workspace_size = max_workspace_size
    else:
        config.max_workspace_size = 10<<30 # 10GB

    parser = trt.OnnxParser(network, logger)                                                         # create parser
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        # import pdb; pdb.set_trace()
        (onnx_path, _) = os.path.split(onnxFile)
        if not parser.parse(model.read(), path=onnxFile):                                            # parse onnx
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
    print("Succeeded parsing ONNX file!")

    if use_fp16:                                                                                      # init config
        config.set_flag(trt.BuilderFlag.FP16)
        plan_name = plan_name.replace(".plan", "_fp16.plan")

    if builder_opt_evel:
        config.builder_optimization_level = builder_opt_evel

    # set profile
    assert network.num_inputs == len(min_shapes)
    assert network.num_inputs == len(opt_shapes)
    assert network.num_inputs == len(max_shapes)

    profile = builder.create_optimization_profile()                                                   # create profile
    for i in range(network.num_inputs):                                                               # set profile
        input_tensor = network.get_input(i)
        profile.set_shape(input_tensor.name, tuple(min_shapes[i]), tuple(opt_shapes[i]), tuple(max_shapes[i]))

    config.add_optimization_profile(profile)                                                          # init config

    engine = builder.build_engine(network, config)                                                    # create engine
    if not engine:
        raise RuntimeError("build_engine failed")
    print("Succeeded building engine!")

    print("Serializing Engine...")
    serialized_engine = engine.serialize()                                                             # serialize engine
    if serialized_engine is None:
        raise RuntimeError("serialize failed")

    (plan_path, _) = os.path.split(plan_name)
    os.makedirs(plan_path, exist_ok=True)
    with open(plan_name, "wb") as fout:
        fout.write(serialized_engine)


def export_yolov5_model():
    onnx_path = "./onnx/3_6_best.onnx"
    plan_path = "./engine/3_6_best.plan"

    # onnx2trt(onnx_path, plan_path,
            # [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)])

    onnx2trt(onnx_path, plan_path,
            [(1, 3, 1120, 1120)], [(1, 3, 1120, 1120)], [(1, 3, 1120, 1120)],
             use_fp16=False)

    print("======================= Decoder  onnx2trt done!")



def main():
    export_yolov5_model()


if __name__ == '__main__':
    main()
