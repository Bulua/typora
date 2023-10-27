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

在 OpenCV 中，`VideoWriter_fourcc` 是用于设置视频编码格式的函数。`fourcc` 代表 "four character code"，是一个四字符编码，用于指定视频编码器的标识符。不同的编码器有不同的 fourcc。

以下是一些常见的视频编码器及其对应的 fourcc：

1、**XVID 编码器:**

```
pythonCopy code
fourcc = cv2.VideoWriter_fourcc(*'XVID')
```

2、**H.264 编码器:**

```
pythonCopy code
fourcc = cv2.VideoWriter_fourcc(*'H264')
```

3、**MJPG 编码器:**

```
pythonCopy code
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
```

4、**MP4V 编码器:**

```
pythonCopy code
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
```
