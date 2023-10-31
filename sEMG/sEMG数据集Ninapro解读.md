[toc]



## 1、数据集介绍

做`sEMG`信号的常用数据集是`NinaPro`（http://ninapro.hevs.ch/），目前包含`DB[1-10]`个完整数据集，其中包含截肢者和非截肢者的`sEMG`信号文件。所有数据集的描述都类似，熟悉单个数据集的数据组成即可，该文主要以`NinaproDB1`数据集来介绍。


## 2、NinaproDB1

该 `DB1`数据集包括 `27 `名完整受试者重复 `52 `次**手部动作**以及**休息位置时**的**表面肌电图**和**运动学数据**。

![20231031144830](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231031144830.png)

## 3、采集设备

`sEMG `数据使用 `10 `个 `Otto Bock MyoBock 13E200 `电极获取，而运动学数据使用 `Cyberglove 2` 数据手套获取。

<img src="https://ninapro.hevs.ch/figures/setup_cyberglove.png" style="zoom:50%;" />

## 4、采集动作

该数据集包括 `52 `个不同动作的` 10 `次重复。受试者被要求重复在笔记本电脑屏幕上以电影形式显示的动作。

实验分为三个练习： 

- 手指的基本动作。 

- 等距、等张的手部配置和基本的手腕运动。 

- 抓握和功能性动作。

<img src="https://ninapro.hevs.ch/figures/SData_Movements.png" style="zoom: 50%;" />

## 5、文件数据描述

下载其中`s1`受试者的数据文件，解压后：
![20231031145026](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231031145026.png)

可以使用`matlab`或者`python`来读取文件内容。使用matlab直接双击`mat`文件就可以在变量区获得数据了，这里仅演示用`python`来读取文件数据。

```python
# 安装所需要的库：pip install scipy

import scipy.io as scipy

path = '你的mat文件路径'
file = scipy.loatmat(path)

# file是个字典类型，查看所有的键值
print(file.keys())

# 输出：
dict_keys(['__header__', '__version__', '__globals__', 'emg', 'stimulus', 'glove', 'subject', 'exercise', 'repetition', 'restimulus', 'rerepetition'])
```

接下来解释每个键的含义：

1. ``__header__`：文件头信息，包含文件创建时间等信息。
2. `__version__`：文件版本信息。
3. `__globals__`：为空，对后续实验没影响。
4. `emg`：是一个二维矩阵，横坐标是时间戳，纵坐标代表通道，`S1_A1_E1.mat`的`emg`数据形状为`(101014, 10)`，第 `1-8` 列是在桡肱关节高度处围绕前臂等距分布的电极。第` 9` 列和第` 10 `列包含来自指浅屈肌和伸肌的主要活动点的信号。
5. `stimulus`：是一个二维矩阵，横坐标是时间戳，纵坐标仅有一列，值代表受试者根据所显示的电影重复的动作。
6. `glove`：是一个二维矩阵，横坐标是时间戳，纵坐标代表（`22`个）通道。网络手套 22 个传感器的未校准信号。
7. `subject`：受试者编号。
8. `exercise`：训练编号，前面说了`DB1`共有三个训练，这个就是三个训练的编号。
9. `repetition`：是一个二维矩阵，横坐标是时间戳，纵坐标仅有一列。代表每个时间戳所对应手势重复执行的次数，最大为`10`，最小为`0`，`0`表示休息阶段。
10. `restimulus`：是一个二维矩阵，横坐标是时间戳，纵坐标仅有一列。每个值表示训练`A、B、C`的手势编号。具有事后细化的运动标签的持续时间，以便更好地对应于真实的运动。==可以作为手势分类标签来用==。
11. `rerepetition`：与`repetition`有相似的含义，是一个二维矩阵，横坐标是时间戳，纵坐标仅有一列。代表每个时间戳所对应手势重复执行的次数，最大为`10`，最小为`0`，`0`表示休息阶段。

## 6、数据处理

### 6.1 获取数据

```python
data = file['emg']      # (times, channel)
label = file['restimulus']  # (times, 1)
```

### 6.2 数据处理

`data`中包含很多休息段的`semg`数据，我们可以根据`label`来获取休息段的下标值，之后根据下标值来获取`data`中所有活动段的数据。

```python
active_index = (label != 0).flatten()
data = data[active_index]
label = label[active_index]
```

## 7、踩坑

### 7.1 标签无法对应的问题

有数据集的文件存在手势标签不对的情况，这里建议每次读取`label`后，打印一下包含哪类标签值。训练`A、B、C`和`D`文件的标签一定要对应上手势数量，需要进行后面的处理。

```python
unique_labels = np.unique(label)
print(unique_labels)

# 由于我读取的是训练1的文件，因此包含12个手势标签，这里是没问题的
# 输出：
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=uint8)
```

### 7.2 restimulus键缺失

使用`stimulus`代替即可。


之后怎么处理可以根据自己的实验来进行，如果需要帮助可在评论区留言，记得点个关注哦😀！
