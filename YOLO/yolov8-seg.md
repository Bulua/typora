[toc]

# 该文件包含所有关于Yolo-seg操作的实例

## 1、从图片中切割出识别的物体

### 1.1 路径、资源准备

准备一张图片，并在[链接](https://docs.ultralytics.com/models/yolov8/#supported-modes)中查找并下载权重文件

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.augment import LetterBox

img = 'p1.png'
model = YOLO('yolov8m-seg.pt') # yolov8的权重
```
![20231010114612](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010114612.png)

### 1.2 获取分割结果

```python
results = model(img, classes=[0]) # results包含所有检测出的人物，类别classes根据自己需要更改
r = results[0] # 我们取第一个检测结果
```

### 1.3 提取分割出的物体

```python
def get_mask(result, im_gpu=None, pred_boxes=None):
    annotator = Annotator(deepcopy(result.orig_img), line_width=3, font_size=18)

    pred_masks = result.masks
    if im_gpu is None:
        img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
        im_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
            2, 0, 1).flip(0).contiguous() / 255
    idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
    annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

    return annotator.result()

plt.imshow(get_mask(r))
```
![20231010114455](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010114455.png)

### 1.4 去掉背景

```python
def plot_img(img, masks):
    pred_masks = masks.data.cpu().numpy()
    pred_masks = np.concatenate([pred_masks, pred_masks, pred_masks])
    pred_masks = np.transpose(pred_masks, (1, 2, 0))
    crop_img = (pred_masks * img)
    crop_img = crop_img.astype(np.uint8)    # 数据格式转换为可显示的uint类型
    plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    return crop_img

crop_img = plot_img(r.orig_img, r.masks)
```

![20231010114400](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010114400.png)

























