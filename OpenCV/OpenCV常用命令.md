[toc]

# OpenCV常用命令

## 1、读取视频

```python
cap = cv2.VideoCapture(video_path)
```

## 2、获取视频的参数

```python
# 获取视频的总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 获取视频的高度
height =  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 获取视频的宽度
weight =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 获取视频的帧率
fps    =  int(cap.get(5))
```

### 3、写入视频

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(video_save_path, fourcc, fps, (w, h))
```
