import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import cv2
import datetime
from share import *
import config

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

"""
本文档是课程的作业1答案，区别于用于实际部署的代码。
它会包含大量的个人注释，涉及相关知识点和思考，便于我解答学生的问题。
"""


#? 实际价值多大呢？是否可以对CLIP模型以外的其他模型做一样的处理呢？
# 实测：482756KB -> 482203KB 较少了几百KB；
def optimize(onnx_path, opt_onnx_path):
    from onnxsim import simplify
    # onnxsim.simplify 做的事情是通过推理整个计算图，用常量输出替换冗余操作符（常量折叠）来简化模型，减少模型中节点数量，使模型结构更加清晰；
    # https://www.dongaigc.com/a/onnx-simplifier-simplify-your-onnx-model
    # 简化方式包含：移除冗余节点；融合节点；常量折叠；去除不必要操作；消除不必要的输入和输出；简化模型结构
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    print(f"{onnx_path} simplify start !")
    # print("init", graph)
    model_simp, check = simplify(model)
    # print("opt", gs.import_onnx(model_simp))                        # 打印优化后的模型结构，CLIP这么大的模型实在是不需要查看。
    onnx.save(model_simp, opt_onnx_path, save_as_external_data=False)  
    #onnx.save(model_simp, opt_onnx_path, save_as_external_data=True) # 它能够将模型权重和模型结构分开存储；
    # 好处主要是可以减小模型文件的大小，提高模型的加载速度(加载可以并行执行)；
    # 缺点是使用外部数据可能会增加模型部署的复杂性。因为需要额外的步骤读取外部数据文件。
    
    assert check, "Simplified ONNX model could not be validated"
    print(f"{onnx_path} simplify done !")
    
    #? onnx模型中除了要了解其三个组成元素以外，还需要知道onnx的节点名称代表什么意思；
    #- 比如gather算子；data，dim，indices




def onnxruntime_check(onnx_path, input_dicts, torch_outputs):
    onnx_model = onnx.load(onnx_path)          # 1 加载模型
    # onnx.checker.check_model(onnx_model)     # 检查模型
    sess = rt.InferenceSession(onnx_path)      # 2 初始化会话对象
    # outputs = self.get_output_names()
    # latent input
    # data = np.zeros((4, 77), dtype=np.int32)
    result = sess.run(None, input_dicts)       # 3 执行推理
    

    

    for i in range(0, len(torch_outputs)):
        model_trt_outputs = result[i]
        model_torch_outputs = torch_outputs[i].detach().numpy()
        
        ret = np.allclose(model_trt_outputs, model_torch_outputs, rtol=1e-03, atol=1e-05, equal_nan=False)
        
        if ret is False:
            print("[ERROR] onnxruntime_check")
            # 检测result[i]的输出中是否包含nan
            if np.isnan(model_trt_outputs).any():   print("model_trt_outputs.nan")
            if np.isnan(model_torch_outputs).any(): print("model_torch_outputs.nan")
        else:
            print("onnxruntime_check [PASS]")
        
        # 检查result[i]和torch_outputs[i].detach().numpy()的内容
        print("[Output] model_trt_outputs   :", np.round(np.sum(model_trt_outputs), 6))
        print("[Output] model_torch_outputs :", np.round(np.sum(model_torch_outputs), 6))
        
        # 检查result[i]和torch_outputs[i].detach().numpy()权重和的差值
        output_sum_diff = np.abs(np.round(np.sum(model_trt_outputs) - np.sum(model_torch_outputs), 6))
        print("[Output_Sum_Diff] (|model_trt - model_torch|) :", output_sum_diff)
        
        
        # 检查result[i]和torch_outputs[i].detach().numpy()的形状
        print("[Shape] model_trt: {};  model_torch: {};".format(model_trt_outputs.shape, model_torch_outputs.shape))
        print("-----------------------------------------")
        
        
    input("onnxruntime_check done! details are shown above! \npress any key to continue...")
    
        
class hackathon():
    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cpu'))
        # self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cpu()
        self.model.eval()
        self.ddim_sampler = DDIMSampler(self.model)

hk = hackathon()
hk.initialize()

def export_clip_model():
    clip_model = hk.model.cond_stage_model  
    #? 从某种程度上需要知道为啥选择self.model的cond_stage_model，
    # 其实这个模型实际上是FrozenClipEmbdding模型的实例化对象，包含了tonkenizer和transoformer模型；


    
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

    clip_model.forward = types.MethodType(forward, clip_model)   
    # 这里将clip_model的默认执行函数encode函数替换为forward函数；
    #? 为什么不能将tokenizer模型进行ONNX转换呢？
    # 主要原因是 tokenizer 通常是一个预处理步骤，它将文本输入转换为模型可以理解的格式（例如，将文本转换为整数索引）。这个过程通常不涉及复杂的计算，因此不需要使用 ONNX 进行优化。
    # 它将文本分割成一个个的token，构建词汇表，token映射ID，将文本序列转换成整数索引序列，有时为了确保所有输入序列长度相同，会在短序列后面添加特殊的pad token，或者进行截断。
    # tokenizer 负责将输入的文本转换为整数索引，而 transformer 模型则负责将这些整数索引转换为嵌入向量

    onnx_path = "./onnx/CLIP.onnx"

    tokens = torch.zeros(1, 77, dtype=torch.int32)
    # 77是每个输入序列的长度，NLP中通常会将本文序列转换成固定长度的整数索引序列，如果样本长度小于77会填充，如果超过则进行截断。
    # 77是一个经验值，通常需要根据实际需求和模型特点进行调整。
    
    
    input_names = ["input_ids"]
    output_names = ["last_hidden_state"]

    torch.onnx.export(
        clip_model,
        (tokens),
        onnx_path,
        verbose=True,
        opset_version=18,
        do_constant_folding=True,    # 模型优化操作：常量折叠！牛批
        input_names=input_names,
        output_names=output_names,
    )
    print("======================= CLIP model export onnx done!")

    # verify onnx model
    output = clip_model(tokens)
    input_dicts = {"input_ids": tokens.numpy()}
    onnxruntime_check(onnx_path, input_dicts, [output])
    
    print("======================= CLIP onnx model verify done!")
    
    torch_path = "./torch_model/CLIP.pt"
    torch.save(clip_model, torch_path)

    # opt_onnx_path = "./onnx/CLIP.opt.onnx"
    # optimize(onnx_path, opt_onnx_path)
    # 实测：482756KB -> 482203KB 较少了几百KB；
    
    torch_path = "./torch_model/CLIP.pt"
    torch.save(clip_model, torch_path)
    


def export_control_net_model():
    control_net = hk.model.control_model

    #! 这四个输出分别从哪里来的？
    # 1. x_noisy 来自于 DDIMSampler.sample() 函数的 x_T 变量；
    # 2. hint 来自于 CannyDetector() 函数的 detected_map 变量；
    #? 3. timestep 来自于 DDIMSampler.sample() 函数的 timesteps 变量；就是batchsize
    # 4. context 来自于 FrozenCLIPT5Encoder() 函数的 encode() 函数的输出；
    x_noisy  = torch.randn(1, 4, 32, 48, dtype=torch.float32)    #! shape: (1, 4, 32, 48) 代表啥意思 -> 代表输入图像的特征图；为啥是4？
    hint     = torch.randn(1, 3, 256, 384, dtype=torch.float32)  # shape: (1, 3, 256, 384) 代表啥意思 -> 代表输入图像的边缘图；
    timestep = torch.tensor([1], dtype=torch.int32)              # shape: (1,) 代表啥意思 -> 代表输入图像的时间步；
    context  = torch.randn(1, 77, 768, dtype=torch.float32)      # shape: (1, 77, 768) 代表啥意思 -> 代表输入图像的文本特征；

    input_names = ["x_noisy", "hint", "timestep", "context"]
    output_names = ["latent"]

    onnx_path = "./onnx/ControlNet.onnx"

    torch.onnx.export(
        control_net,
        (x_noisy, hint, timestep, context),
        onnx_path,
        verbose=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        keep_initializers_as_inputs=True   #! 为什么要设置为True？
                                           #! 为什么不设置output_names？
    )

    # import pdb; pdb.set_trace()
    outputs = control_net(x_noisy, hint, timestep, context)
    print("=== {}".format(len(outputs)))
    # outputs包含哪些内容？13个！

    input_dicts = {"x_noisy": x_noisy.numpy(), "hint": hint.numpy(), "timestep": timestep.numpy(), "context": context.numpy()}
    onnxruntime_check(onnx_path, input_dicts, outputs)
    
    
    torch_path = "./torch_model/ControlNet.pt"
    torch.save(control_net, torch_path)
    
    


def export_controlled_unet_model():
    controlled_unet_mdoel = hk.model.model.diffusion_model

    #? controlnet 和 unet 的联系？为什么会有相似的输入？
    # controlnet: {"x_noisy":img, "hint":hint, "timestep":ts, "context":cond_txt}
    # unet:       {'x_noisy': img, 'timestep': ts, 'context': cond_txt, 'control0': control[4], 'control1': control[5], 'control2': control[6], 'control3': control[7], 'control4': control[8], 'control5': control[9], 'control6': control[10], 'control7': control[11], 'control8': control[12], 'control9': control[13], 'control10': control[14], 'control11': control[15], 'control12': control[16]}
    
    x_noisy = torch.randn(1, 4, 32, 48, dtype=torch.float32)
    timestep = torch.tensor([1], dtype=torch.int32)
    context = torch.randn(1, 77, 768, dtype=torch.float32)

    # control 为一个list 里面为tensor 13个
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

    input_names = ["x_noisy", "timestep", "context"]
    for i in range(0, len(control_list)):
        input_names.append("control" + str(i))

    output_names = ["latent"]

    #! 为什么ControlledUnet夹下存在大量的权重文件？
    # 事实上真正的onnx模型很小，大部分是权重文件
    onnx_path = "./onnx/ControlledUnet"
    os.makedirs(onnx_path, exist_ok=True)
    onnx_path = onnx_path + "/ControlledUnet.onnx"

    torch.onnx.export(
        controlled_unet_mdoel,
        (x_noisy, timestep, context, control_list),
        onnx_path,
        verbose=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )

    # verify onnx model
    input_dicts = {"x_noisy": x_noisy.numpy(), "timestep": timestep.numpy(), "context": context.numpy()}
    for i in range(0, len(control_list)):
        input_dicts["control" + str(i)] = control_list[i].numpy()

    # TODO: controlled_unet_mdoel will make control_list = [] 所以呢？
    # 其他使用control_list的操作(比如input_dicts对control_list的使用)需要放置在controlled_unet_mdoel执行的前面
    output = controlled_unet_mdoel(x_noisy, timestep, context, control_list)

    onnxruntime_check(onnx_path, input_dicts, [output])
    
    
    torch_path = "./torch_model/ControlledUnet.pt"
    torch.save(controlled_unet_mdoel, torch_path)


def export_decoder_model():
    # control_net = hk.model.control_model

    decode_model = hk.model.first_stage_model
    decode_model.forward = decode_model.decode  # decode方法中包含了encoder模型、conv模型和DiagonalGaussianDistribution模型

    latent = torch.randn(1, 4, 32, 48, dtype=torch.float32)

    input_names = ["latent"]                    # 输入名称最好与controlnet和unet的输出名称保持一致
    output_names = ["images"]

    onnx_path = "./onnx/Decoder.onnx"

    # import pdb; pdb.set_trace()
    ret = decode_model(latent)

    torch.onnx.export(
        decode_model.cpu(),
        (latent),
        onnx_path,
        verbose=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        keep_initializers_as_inputs=True
    )

    input_dicts = {"latent": latent.numpy()}
    onnxruntime_check(onnx_path, input_dicts, [ret])
    
    torch_path = "./torch_model/Decoder.pt"
    torch.save(decode_model, torch_path)

def main():
    #? 以下模型的导出过程需要基于torch源码而不是trt源码进行！否则unet的导出存在问题。
    # 事实上，除了decoder模型其他模型的torch模型大小和trt模型大小基本一致。
    
    # export_clip_model()
    # export_control_net_model()
    export_controlled_unet_model()
    # export_decoder_model()
    
    # onnx_path = "./onnx/CLIP.onnx"
    # opt_onnx_path = "./onnx/CLIP.opt.onnx"
    # optimize(onnx_path, opt_onnx_path)
    
    pass

if __name__ == '__main__':
    main()
