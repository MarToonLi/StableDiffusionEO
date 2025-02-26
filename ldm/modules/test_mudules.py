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
            print("qkv_w: ",self.qkv_w.sum())
            
        else:
            self.kv_w = torch.cat([self.to_k.weight, self.to_v.weight]).transpose(0, 1).detach()
            print("kv_w: ",self.kv_w.sum())
            
        # =============================

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        # 输出to_q层的网络权重的总和值
        print(self.to_q.weight.sum())
        print(self.to_k.weight.sum())
        print(self.to_v.weight.sum())

        # add content =================
        if context is None:
            print(1)
            qkv = torch.matmul(x, self.qkv_w)
            q, k, v = qkv.chunk(3, dim=-1)
            
            print("conextNone:  q:{}; k:{}; v:{};".format(q.sum(), k.sum(), v.sum()))
            
        else:
            print(2)
            q = self.to_q(x)
            kv = torch.matmul(context, self.kv_w)
            k, v = kv.chunk(2, dim=-1)
            print("conext not None:  q:{}; k:{}; v:{};".format(q.sum(), k.sum(), v.sum()))
            
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


class CrossAttention_beifen(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        # 输出to_q层的网络权重的总和值
        print(self.to_q.weight.sum())
        print(self.to_k.weight.sum())
        print(self.to_v.weight.sum())

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        print("beifen:  q:{}; k:{}; v:{};".format(q.sum(), k.sum(), v.sum()))
        

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



# 定义模型参数
query_dim = 512  # 假设的query维度
heads = 8        # 多头注意力头数
dim_head = 64    # 每个头的维度
context_dim = 77
# 初始化模型
model = CrossAttention(query_dim=query_dim, heads=heads, dim_head=dim_head, context_dim = context_dim)
model_beifen = CrossAttention_beifen(query_dim=query_dim, heads=heads, dim_head=dim_head, context_dim = context_dim)

# 同步权重
# model_beifen.load_state_dict(model.state_dict())

# 定义测试输入（batch_size=2, seq_len=10, query_dim=512）
torch.manual_seed(0)  # 固定随机种子
x = torch.randn(2, 10, query_dim)
content = torch.randn(2, 10, context_dim) if context_dim is not None else None

# 前向对比
with torch.no_grad():
    out1 = model(x, content)
    out2 = model_beifen(x, content)
    
    
    print(out1.sum(), out2.sum())
    print(out1.shape)
    print(f"输出形状是否一致: {out1.shape == out2.shape}")  # 预期True
    print(f"张量是否相近: {torch.allclose(out1, out2, atol=1e-6)}")  # 预期True
    print(f"最大绝对误差: {torch.max(torch.abs(out1 - out2)):.6f}")  # 预期极小值
