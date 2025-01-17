from Engine_yolov5 import Engine
import os, cv2
import torch
import time
import onnxruntime as rt
import numpy as np
from misc import PreProcessor, PostProcessor, draw_boxes



def check_onnx_trt_outputs(outputs_onnx, outputs_trt, posted_result_onnx: list, posted_result_trt: list):
    for i in range(0, len(outputs_onnx)):  # i  batchsize
        model_trt_outputs  = outputs_trt[i]
        model_onnx_outputs = outputs_onnx[i]
        posted_trt_results = posted_result_trt[i]
        posted_onnx_results = posted_result_onnx[i]
        if isinstance(posted_trt_results, torch.Tensor): posted_trt_results = posted_trt_results.cpu().numpy()
        if isinstance(posted_onnx_results, torch.Tensor): posted_onnx_results = posted_onnx_results.cpu().numpy()
        
        ret = np.allclose(model_trt_outputs, model_onnx_outputs, rtol=1e-03, atol=1e-05, equal_nan=False)
        
        if ret is False:
            print("[ERROR] onnxruntime_check")
            # 检测result[i]的输出中是否包含nan
            if np.isnan(model_trt_outputs).any():   print("model_trt_outputs.nan")
            if np.isnan(model_onnx_outputs).any():  print("model_onnx_outputs.nan")
        else:
            print("onnxruntime_check [PASS]")
        
        print(model_trt_outputs[0])
        print(model_onnx_outputs[0])
        
        # 检查result[i]和torch_outputs[i].detach().numpy()的内容
        print("[Output] model_trt_outputs   :", np.round(np.sum(posted_trt_results), 6))
        print("[Output] model_onnx_outputs :",  np.round(np.sum(posted_onnx_results), 6))
        
        # 检查result[i]和torch_outputs[i].detach().numpy()权重和的差值
        output_sum_diff = np.abs(np.round(np.sum(model_trt_outputs) - np.sum(model_onnx_outputs), 6))
        print("[Output_Sum_Diff] (|model_trt - model_onnx|) :", output_sum_diff)
        
        
        # 检查result[i]和torch_outputs[i].detach().numpy()的形状
        print("[Shape] model_trt: {};  model_onnx: {};".format(model_trt_outputs.shape, model_onnx_outputs.shape))
        print("-----------------------------------------")









image_path = r"/data/ex_space/datasets/3-6/NG_bootDent_0810 (233).png"
clip_engine_path = "/data/Projects/StableDiffusionEO/engine/3_6_best.plan"
onnx_path        = "/data/Projects/StableDiffusionEO/onnx/3_6_best.onnx"
save_img_path_trt  = "./practice_yolov5/test_trt.jpg"
save_img_path_onnx = "./practice_yolov5/test_onnx.jpg"
model_inputSize = (1120, 1120)
conf_thresh = 0.9
nms_thresh = 0.5
onnx_names = {"input": "images", "output": "output0"}

path = "/data/Projects/StableDiffusionEO/practice_yolov5/trt_yolov5.py"
parent_directory = os.path.dirname(path)


## ---------- 图像读取 ----------
origin_img = cv2.imread(image_path)
if origin_img is not None: print("[ReadImage] 正常读取图像; shape: {};".format(origin_img.shape))
else: print("[ReadImage] 无法读取图像，请检查文件路径是否正确。")




## ---------- 数据预处理 ----------
prepocess = PreProcessor(inpHeight=model_inputSize[0], inpWidth=model_inputSize[0])
x, hw_scale, left, top = prepocess(origin_img)
image_tensor = x[None, :, :, :]  # add bs
image_tensor = torch.from_numpy(image_tensor)
print("[prepocess]: preprocess data.shape: {}; model_input.shape:{}".format(x.shape, image_tensor.shape))






## ---------- trt模型配置和ONNX模型配置 ----------
# trt模型推理
clip_engine = Engine(clip_engine_path)
clip_engine.load()
model_feed_dict = clip_engine.yolov5_model_shape_dict()
clip_engine.activate()
clip_engine.allocate_buffers(model_feed_dict)
print("clip_engine context load")
clip_engine.get_engine_infor()

# onnx模型推理
input_dicts = {onnx_names["input"]:image_tensor.numpy()}
sess = rt.InferenceSession(onnx_path)      



## ---------- TRT模型推理和ONNX模型推理 ----------
start_time = time.time()
outputs_trt = clip_engine.infer({onnx_names["input"]:image_tensor})[onnx_names["output"]].cpu().detach().numpy()
end_time = time.time()
print("[Infer TRT] shape:{};".format(outputs_trt.shape))
print("[Infer TRT] time :{}".format(end_time - start_time))

start_time = time.time()
outputs_onnx = sess.run(None, input_dicts)[0]       
end_time = time.time()
print("[Infer ONNX] shape:{};".format(outputs_onnx.shape))
print("[Infer ONNX] time :{}".format(end_time - start_time))







## ---------- post process ----------
# postprocessor
ratioh = origin_img.shape[0] / prepocess.inpHeight
ratiow = origin_img.shape[1] / prepocess.inpWidth
postprocess = PostProcessor(conf_thresh=conf_thresh, nms_thresh=nms_thresh, left = left, top =top, ratioh=ratioh,ratiow=ratiow)
posted_result_trt = postprocess(outputs_trt)
posted_result_onnx = postprocess(outputs_onnx)


posted_result_trt_img0  = posted_result_trt[0]
posted_result_onnx_img0 = posted_result_onnx[0]
print("[Postprocess  TRT]: ", posted_result_trt_img0.shape)
print("[Postprocess  TRT]: ", posted_result_trt_img0)
print("[Postprocess ONNX]: ", posted_result_onnx_img0.shape)
print("[Postprocess ONNX]: ", posted_result_onnx_img0)


## ---------- draw_boxes ----------
if save_img_path_trt  is not None and os.path.exists(os.path.dirname(path)): 
    draw_boxes(origin_img, posted_result_trt_img0,  save_path = save_img_path_trt)
if save_img_path_onnx is not None and os.path.exists(os.path.dirname(path)): 
    draw_boxes(origin_img, posted_result_onnx_img0, save_path = save_img_path_onnx)




## ---------- 检查结果 ----------
check_onnx_trt_outputs(outputs_onnx, outputs_trt, posted_result_onnx, posted_result_trt)

    