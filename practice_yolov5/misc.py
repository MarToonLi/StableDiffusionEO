import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import cv2
import math
import numpy as np
from copy import deepcopy
from thop import profile

# -------------------------- tools ---------------------------
## compute FLOPs & Parameters
def compute_flops(model, img_size, device):
    x = torch.randn(1, 3, img_size, img_size).to(device)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))


# ---------------------------- NMS ----------------------------
## basic NMS
def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

## class-agnostic NMS 
def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

## class-aware NMS 
def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep = np.zeros(len(bboxes), dtype=np.int32)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)      
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)


# ---------------------------- Processor for Deployment ----------------------------
## Pre-processer
class PreProcessor(object):
    def __init__(self, inpHeight, inpWidth):
        self.inpHeight = inpHeight
        self.inpWidth = inpWidth
        self.input_size = [inpHeight, inpWidth]
        

    def __call__(self, image, swap=(2, 0, 1)):
        """
        Input:
            image: (ndarray) [H, W, 3] or [H, W]
            formar: color format
        """
        print("[PreProcess] image.shape:{}".format(image.shape))
        if len(image.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3), np.float32) * 114.
        elif len(image.shape) == 2:
            padded_img = np.ones(self.input_size, np.float32) * 114.
            
        
        
        # resize and padding
        srch, srcw = image.shape[:2]
        hw_scale = srch / srcw
        
        left = 0
        top = 0
        
        if hw_scale > 1:
            newh = self.inpHeight
            neww = int(self.inpWidth / hw_scale)
            resize_size = (neww, newh)
            resized_img = cv2.resize(image, resize_size, interpolation=cv2.INTER_AREA)
            left = int((self.inpWidth - neww) * 0.5)
            padded_img = cv2.copyMakeBorder(resized_img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        else:
            newh = int(self.inpHeight * hw_scale)
            neww = self.inpWidth
            resize_size = (neww, newh)
            resized_img = cv2.resize(image, resize_size, interpolation=cv2.INTER_AREA)
            top = int((self.inpHeight - newh) * 0.5)
            padded_img = cv2.copyMakeBorder(resized_img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # transpose 
        padded_img = padded_img.transpose(swap) # [H, W, C] -> [C, H, W]
        
        # normalize
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.

        return padded_img, hw_scale, left, top



## Post-processer
class PostProcessor(object):
    def __init__(self, conf_thresh, nms_thresh, left, top, ratiow, ratioh):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.top = top
        self.left = left
        self.ratiow = ratiow
        self.ratioh = ratioh


    def __call__(self, prediction):
        """
        Input:
            prediction: (ndarray) [bs, n_anchors_all, 4+1+C]
        Output:
            outputs: (ndarray) [boxes_number, (x,y,x,y, cls_conf(scores), cls(label))]
        """
        
        
        if not isinstance(prediction, torch.Tensor): prediction = torch.from_numpy(prediction)
        outputs = self.non_max_suppression(prediction, self.conf_thresh, self.nms_thresh)
        
        return outputs
    
    def xywh2xyxy(self, x, ratiow, ratioh,padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] - x[:, 2] / 2) * ratiow  # top left x
        y[:, 1] = (x[:, 1] - x[:, 3] / 2) * ratioh  # top left y
        y[:, 2] = (x[:, 0] + x[:, 2] / 2) * ratiow  # bottom right x
        y[:, 3] = (x[:, 1] + x[:, 3] / 2) * ratioh  # bottom right y
        return y


    def non_max_suppression(
        self,
        prediction,            # [batchsize, anchors, 4+1+nc]
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
    ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
        Inputs:
            prediction: (ndarray) [batchsize, anchors, 4+1+nc]
        Returns:
            list of detections, on (n,6) tensor per image (xyxy, cls_conf(scores), cls(label))
        """
        # basic
        bs = prediction.shape[0]              # batch size
        nc = prediction.shape[1] - 5          # number of classes
        xc = prediction[..., 4]               # candidates
        
        
        # Settings
        xc = xc > conf_thres
        mi = 5 + nc                          # mask start index
        output = [torch.zeros((0, 6))] * bs
        
        
        # 遍历每个batch
        for xi, x in enumerate(prediction):  # image index, image inference [anchors, 4+1+nc]
            
            #? Filter by obj_conf
            x = x[xc[xi]]                    # confidence (filter)

            #? If none remain process next image
            if not x.shape[0]: continue

            # settings
            x[:, 5:] *= x[:, 4:5]      # conf = obj_conf * cls_conf
            box = self.xywh2xyxy(x[:, :4], ratiow=self.ratiow, ratioh=self.ratioh,padw=self.left, padh=self.top)  
            # center_x, center_y, width, height) to (x1, y1, x2, y2)

            #? Filter by cls_conf
            #? Detections matrix nx6 (xyxy, cls_conf(scores), cls(label))
            cls_conf, j = x[:, 5:mi].max(1, keepdim=True)  # conf:(N, 1)，表示每个边界框的最大类别置信度分数。j:(N, 1)，表示每个边界框的最大类别置信度分数对应的类别索引。
            x = torch.cat((box, cls_conf, j.float()), 1)[cls_conf.view(-1) > conf_thres]   #? 这里不再包含前景置信度

            #? Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes)).any(1)]

            # Check shape
            n = x.shape[0]     # number of boxes
            
            #? Filter by NMS
            if not n:continue  # no boxes
            else: x = x[x[:, 4].argsort(descending=True)]      # sort by confidence
            boxes, scores = x[:, :4], x[:, 4]                  # boxes, scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            
            # Output
            output[xi] = x[i]
            
        return output
    


def draw_boxes(image, predcitions, class_names=None, save_path=None):
    """
    将NMS的输出信息标注到图像上，并将标注后的图像保存到指定的地址。

    参数:
        image: 输入的图像
        predcitions: NMS输出的边界框信息，形状为 [num_boxes, 6]，每个边界框包含的信息是 [x1, y1, x2, y2, score, label_id]
        class_names: 类别名称列表
        save_path: 标注后的图像保存地址
    """
    if isinstance(predcitions, torch.Tensor): predcitions = predcitions.cpu().numpy()
    
    # 复制图像，避免在原始图像上进行修改
    image = image.copy()

    # 遍历每个边界框
    for i, box_info in enumerate(predcitions):
        # 获取边界框的坐标
        x1, y1, x2, y2 = box_info[:4].astype(int)

        # 获取标签置信度和类别
        confidence = box_info[4].astype(float)
        class_id = box_info[5].astype(int)

        # 获取类别名称
        if class_names is None: class_name = str(class_id)
        else: class_name = class_names[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 在边界框上方绘制标签信息
        label = f'{class_name}: {confidence:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存标注后的图像
    cv2.imwrite(save_path, image)
