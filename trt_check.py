from Engine import Engine
import torch

clip_engine_path = "/data/Projects/StableDiffusionEO/engine/CLIP_work1_float32_opt.plan"
clip_engine = Engine(clip_engine_path)
clip_engine.load()
model_feed_dict = clip_engine.clip_model_shape_dict(1, 77, embedding_dim = 768)
clip_engine.activate()
clip_engine.allocate_buffers(model_feed_dict)
clip_engine.get_engine_infor()
    
tokens = torch.randint(low=0, high=40000, size=(1, 77), dtype=torch.int32)  # torch.onnx.export参数
outputs_trt = clip_engine.infer({"input_ids":tokens})['last_hidden_state'].clone()
print(outputs_trt)