import numpy as np
import cv2
import datetime
from share import *

import cv2
import einops
# import gradio as gr
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


from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # q.shape: inner_dim, query_dim; k.shape: inner_dim, context_dim; v.shape: inner_dim, context_dim;
        
        # add content =================
        if context_dim == query_dim:  # context_dim == None
            self.qkv_w = torch.cat([self.to_q.weight, self.to_k.weight, self.to_v.weight]).transpose(0, 1).detach()
        else:
            self.kv_w = torch.cat([self.to_k.weight, self.to_v.weight]).transpose(0, 1).detach()
        # =============================

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        # add content =================
        if torch.onnx.is_in_onnx_export():
            if context is None:
                print("export: conext None:  x:{}; context:{};".format(x.sum(), context.sum() if context is not None else 0))
                
                qkv = torch.matmul(x, self.qkv_w)
                q, k, v = qkv.chunk(3, dim=-1)
                print("export: conext None:  q:{}; k:{}; v:{};".format(q.sum(), k.sum(), v.sum()))
                # print("export: conext None:  q:{}; k:{}; v:{};".format(self.to_q.weight.sum(), self.to_k.weight.sum(), self.to_v.weight.sum()))
                
            else:
                print("export: conext not None:  x:{}; context:{};".format(x.sum(), context.sum() if context is not None else 0))
                
                q = self.to_q(x)
                kv = torch.matmul(context, self.kv_w)
                k, v = kv.chunk(2, dim=-1)
                print("export: conext not None:  q:{}; k:{}; v:{};".format(q.sum(), k.sum(), v.sum()))
                # print("export: conext not None:  q:{}; k:{}; v:{};".format(self.to_q.weight.sum(), self.to_k.weight.sum(), self.to_v.weight.sum()))
                
                
        else:
            context = default(context, x)
            print("[train]:  x:{}; context:{};".format(x.sum(), context.sum() if context is not None else 0))
            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)
            print("[train]:  q:{}; k:{}; v:{};".format(q.sum(), k.sum(), v.sum()))
            # print("[train]:  conext not None:  q:{}; k:{}; v:{};".format(self.to_q.weight.sum(), self.to_k.weight.sum(), self.to_v.weight.sum()))
            
            
        # =============================
		
        # commented ======================
        # q = self.to_q(x)
        # context = default(context, x)
        # k = self.to_k(context)
        # v = self.to_v(context)
        # =============================

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


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
            
            
            
def export_attention_model():

    # H 输入变量构造
    # 定义模型参数
    query_dim = 512  # 假设的query维度
    heads = 8        # 多头注意力头数
    dim_head = 64    # 每个头的维度
    context_dim = None
    # 初始化模型
    model = CrossAttention(query_dim=query_dim, heads=heads, dim_head=dim_head, context_dim = context_dim)
    
    attNet = model.cpu()
    torch.manual_seed(0)  # 固定随机种子
    x = torch.randn(1, 10, query_dim, dtype=torch.float32)
    content = torch.randn(1, 10, context_dim, dtype=torch.float32) if context_dim is not None else None

    input_names = ["x", "content"] if context_dim is not None else ["x"]
    output_names = ["latent"]   # 2, 10, 512

    # H onnx模型输出
    onnx_path = "./onnx/attNet_work3.onnx"

    torch.onnx.export(
        attNet,
        (x, content),
        onnx_path,
        verbose=True,
        opset_version=18,                      # torch.onnx.export参数
        input_names=input_names,
        output_names=output_names,
    )
    print("======================= ControlNet model export onnx done!")

    input_dicts = {"x": x.numpy(), "content": content.numpy()}  if context_dim is not None else {"x": x.numpy()}
    outputs = model(x, content if context_dim is not None else None)
    onnxruntime_check(onnx_path, input_dicts, outputs)
    print("======================= ControlNet onnx model verify done!")

def main():
    export_attention_model()


if __name__ == '__main__':
    main()