import numpy as np
import cv2
import datetime
from share import *

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from onnx import shape_inference
import onnx_graphsurgeon as gs
import onnx
import onnxruntime as rt
import surgeon_graph

def optimize(onnx_path, opt_onnx_path):
    from onnxsim import simplify
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    print(f"{onnx_path} simplify start !")
    # self.info("init", graph)
    model_simp, check = simplify(model)
    # self.info("opt", gs.import_onnx(model_simp))
    onnx.save(model_simp, opt_onnx_path, save_as_external_data=True)
    assert check, "Simplified ONNX model could not be validated"
    print(f"{onnx_path} simplify done !")


def onnxruntime_check(onnx_path, input_dicts, torch_outputs):
    onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)
    sess = rt.InferenceSession(onnx_path)
    # outputs = self.get_output_names()
    # latent input
    # data = np.zeros((4, 77), dtype=np.int32)
    result = sess.run(None, input_dicts)

    for i in range(0, len(torch_outputs)):
        print(i)
        tmpa = result[i]
        tmpb = torch_outputs[i].detach().numpy()
        print(np.sum(tmpa) - np.sum(tmpb))

        ret = np.allclose(result[i], torch_outputs[i].detach().numpy(), rtol=1e-03, atol=1e-05, equal_nan=False)
        # ATT FP32下 atol 应该是10-6；FP16下应该是10-3；
        # ATT 逐层检验输出：在保证FP32的情况下，逐步尝试FP16和INT8模式！
        if ret is False:
            print("Error onnxruntime_check")
        else:
            print("Yes")


class hackathon():
    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cpu'), strict=False)
        # self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cpu()
        self.model.eval()
        self.ddim_sampler = DDIMSampler(self.model)


hk = hackathon()
hk.initialize()


def export_clip_model():
    print("开始执行clip_model的onnx模型")
    clip_model = hk.model.cond_stage_model  # 获取cond_stage_model模型模块

    import types

    def forward(self, tokens):
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    clip_model.forward = types.MethodType(forward, clip_model)  # 更改torch模块的tensor入口函数

    onnx_path = "./onnx/CLIP_work1.onnx"  # torch.onnx.export参数
    opt_onnx_path = "./onnx/CLIP_work1_float32_opt.onnx"  # torch.onnx.export参数

    tokens = torch.zeros(1, 77, dtype=torch.int32)  # torch.onnx.export参数
    input_names = ["input_ids"]  # torch.onnx.export参数
    output_names = ["last_hidden_state"]  # torch.onnx.export参数
    dynamic_axes = {"input_ids": {1: "S"}, "last_hidden_state": {1: "S"}}  # torch.onnx.export参数
 
    torch.onnx.export(
        clip_model.cpu(),              # 
        (tokens),                      # 输入投入模型，通过trace记录输入走过的所有算子
        onnx_path,
        verbose=True,
        opset_version=18,              # torch.onnx.export参数
        do_constant_folding=True,      # 常量折叠，将计算图中可以在编译时计算的常量表达式进行计算并作为常量节点嵌入到计算图中；属于推荐行为
        input_names=input_names,       #  input_names在模型只有一个输入时可以不显示指明；但是模型输入多个或者希望模型输入名称自定义时除外；不指定则是01234
        output_names=output_names,     # output_names在模型只有一个输出时可以不显示指明；但是模型输出多个或者希望模型输出名称自定义时除外；
        # dynamic_axes=dynamic_axes,     # dynamic_axes需要保证值字典中的值变量能够正确反映实际意义即可，重要的是值字典中的键被正确指明；
                                       # 用于指定在导出的过程中哪些维度应该是动态的，即在运行时可以变化的
                                       # 通常是batch_size，以及NLP中的文本语音中的序列长度；
                                       #!? 如果不指定，则onnx模型的输入张量为固定大小（[3,1,224,224]），无法动态改变；
                                       # 指定后，onnx模型的输入张量shape为[batch_size, 1, 224, 224]
        # keep_initializers_as_inputs=True   # 推荐为False，有利于常数折叠优化；
                                       # 除非特殊要求或者希望模型权重在执行的时候发生更新
    )
    print("======================= CLIP model export onnx done!")
    # surgeon_graph.clip_rm_inf_and_change_inout_type(onnx_path, opt_onnx_path)
    surgeon_graph.clip_rm_inf_and_change_inout_type_int32(onnx_path, opt_onnx_path)

    # verify onnx model
    output = clip_model(tokens)
    input_dicts = {"input_ids": tokens.numpy()}
    onnxruntime_check(opt_onnx_path, input_dicts, [output])
    print("======================= CLIP onnx model verify done!")

    # opt_onnx_path = "./onnx/CLIP.opt.onnx"
    # optimize(onnx_path, opt_onnx_path)


def export_control_net_model():
    control_net = hk.model.control_model.cpu()

    # H 输入变量构造
    x_nosiy = torch.randn(1, 4, 32, 48, dtype=torch.float32)
    hint = torch.randn(1, 3, 256, 384, dtype=torch.float32)
    timestep = torch.tensor([1], dtype=torch.int32)
    context = torch.randn(1, 77, 768, dtype=torch.float32)

    print(x_nosiy.numpy().flatten()[:10])


    input_names = ["x_nosiy", "hint", "timestep", "context"]
    output_names = ["latent"]

    # H onnx模型输出
    onnx_path = "./onnx/ControlNet_work2.onnx"

    torch.onnx.export(
        control_net,
        (x_nosiy, hint, timestep, context),
        onnx_path,
        verbose=True,
        opset_version=18,                      # torch.onnx.export参数
        do_constant_folding=True,
        input_names=input_names,
        keep_initializers_as_inputs=False
    )
    print("======================= ControlNet model export onnx done!")

    input_dicts = {"x_nosiy": x_nosiy.numpy(), "hint": hint.numpy(), "timestep": timestep.numpy(), "context": context.numpy()}
    outputs = control_net(x_nosiy, hint, timestep, context)
    onnxruntime_check(onnx_path, input_dicts, outputs)
    print("======================= ControlNet onnx model verify done!")


def export_controlled_unet_model():
    controlled_unet_mdoel = hk.model.model.diffusion_model.cpu()

    # H 输入变量构造
    x_nosiy = torch.randn(1, 4, 32, 48, dtype=torch.float32)
    timestep = torch.tensor([1], dtype=torch.int32)
    context = torch.randn(1, 77, 768, dtype=torch.float32)

    control_list = [
        torch.randn(1, 320, 32, 48, dtype=torch.float32),
        torch.randn(1, 320, 32, 48, dtype=torch.float32),
        torch.randn(1, 320, 32, 48, dtype=torch.float32),

        torch.randn(1, 320, 16, 24, dtype=torch.float32),
        torch.randn(1, 640, 16, 24, dtype=torch.float32),
        torch.randn(1, 640, 16, 24, dtype=torch.float32),

        torch.randn(1, 640, 8, 12, dtype=torch.float32),
        torch.randn(1, 1280, 8, 12, dtype=torch.float32),
        torch.randn(1, 1280, 8, 12, dtype=torch.float32),

        torch.randn(1, 1280, 4, 6, dtype=torch.float32),
        torch.randn(1, 1280, 4, 6, dtype=torch.float32),
        torch.randn(1, 1280, 4, 6, dtype=torch.float32),
        torch.randn(1, 1280, 4, 6, dtype=torch.float32),

    ]

    input_names = ["x", "timestep", "context"]
    for i in range(0, len(control_list)):
        input_names.append("control" + str(i))
    output_names = ["latent"]

    onnx_path = "./onnx/ControlledUnet_work2"
    os.makedirs(onnx_path, exist_ok=True)
    onnx_path = onnx_path + "/ControlledUnet_work2.onnx"

    torch.onnx.export(
        controlled_unet_mdoel,
        (x_nosiy, timestep, context, control_list),
        onnx_path,
        verbose=True,
        opset_version=18,  # torch.onnx.export参数
        do_constant_folding=True,
        input_names=input_names,
        # output_names=output_names,
        # dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=True,
    )
    print("======================= controlled_unet_mdoel model export onnx done!")

    input_dicts = {"x": x_nosiy.numpy(), "timestep": timestep.numpy(), "context": context.numpy()}
    for i in range(0, len(control_list)):
        print("control" + str(i))
        input_dicts["control" + str(i)] = control_list[i].numpy()
    outputs = controlled_unet_mdoel(x_nosiy, timestep, context, control_list)
    onnxruntime_check(onnx_path, input_dicts, outputs)
    print("======================= controlled_unet_mdoel onnx model verify done!")


def export_decoder_model():
    # control_net = hk.model.control_model

    decode_model = hk.model.first_stage_model
    decode_model.forward = decode_model.decode

    latent = torch.randn(1, 4, 32, 48, dtype=torch.float32)
    input_names = ["latent"]
    output_names = ["images"]
    onnx_path = "./onnx/Decoder_work2.onnx"  

    torch.onnx.export(
        decode_model.cpu(),
        (latent),
        onnx_path,
        verbose=True,
        opset_version=18,                  # torch.onnx.export参数
        do_constant_folding=True,
        input_names=input_names,
        keep_initializers_as_inputs=False   # True时error: 0.0390625; None时error: 0.01171875; False时error: 0.015625;
    )
    print("======================= decode_model model export onnx done!")

    ret = decode_model(latent)
    input_dicts = {"latent": latent.numpy()}
    onnxruntime_check(onnx_path, input_dicts, [ret])
    print("======================= decode_model onnx model verify done!")


def main():
    export_clip_model()
    # export_control_net_model()
    # export_controlled_unet_model()
    # export_decoder_model()


if __name__ == '__main__':
    main()

# https://blog.csdn.net/gulingfengze/article/details/108425949

# https://mmdeploy.readthedocs.io/zh-cn/stable/02-how-to-run/profile_model.html