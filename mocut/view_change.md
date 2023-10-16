[toc]

# 镜头切换检测进展

## 问题描述

由于前后帧可能出现镜头远近切换的情况，导致切片视频包含人物不连续动作的片段。**根据前后帧检测出人物的关键点距离的差异来判断是否出现镜头切换**。

## 1、实现流程

1. 获取前、后帧中人物关键点的位置坐标、关键点置信度。

2. 添加坐标原点（0，0）、坐标原点置信度（为1）。

3. 筛选出置信度较高的关键点，以及关键点的位置坐标。

4. 根据点与点的置信度计算连线的权重$w_{i,j}$，表示点`i`和点`j`连接线的权重。

5. 计算==关键点之间的距离==（用于判断前、后帧人物大小是否相似）、==腰部关键点和坐标原点的距离==（用于判断前、后帧人物位置是否有过大的偏移）。用$d_{i,j}$来表示点`i`到点`j`的距离。

6. 分别求出前、后帧的距离之和$s_1、s_2$。

7. 用以下公式来计算前、后帧人物之间的距离差异$diff$：
   $$
   \begin{aligned}
   s_1 = s_2 &= \sum_{i}^{n}\sum_{j}^{n}w_{i,j}d_{i,j}, \ \ (i,j) \ \in condition \ set		\\
   u &= \frac{s_1}{s_2} - 1 \\
   diff = \text{abs}(tanh) &= \text{abs}(\frac{e^{2u}-1}{e^{2u}+1})
   \end{aligned}
   $$

   其中，$w_{i,j}$和$d_{i,j}$分别表示点`i`和点`j`连接线的权重、距离，n表示关键点的数量（共18个关键点，17个人体上的关键点，1个原点），$condition \ set$代表需要算入总距离的连接线集合（例如肩膀与脚部是不需要算入总距离的，而左肩膀和右肩膀需要算入总距离），$s_1、s_2$分别为前、后帧的距离之和，中间变量$u$则表示$s_1、s_2$之间的差异，而$\tanh$负责将差异$u$缩放到区间`(-1, 1)`之间，`tanh`函数的曲线图像如下所示：

   <img src="https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231012143832.png" alt="20231012143832" style="zoom: 80%;" />

   <center>图 1. tanh函数图</center>

   使用$\tanh$函数可将所有结果缩放到`(-1, 1)`之间，更利于设计合理的阈值进行判断，还可对阈值进行调整，作为判定视频帧变化的标准。$\text{abs}$将输入值变为正数。

8. 当差异`diff`过大时，存在镜头切换，否则不存在。
9. 描述差异是否过大是通过设计一个阈值`th`来判定，目前设定`th=0.10`。较低的`th`代表对画面人物变化的容忍度较低，较高的`th`代表对画面人物变化容忍度较高。

## 2、优化效果

### 2.1 案例1

> 视频中由于镜头切换导致人物位置变化较大

|   **优化前**   | ![otetj-gi6fv](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/otetj-gi6fv.gif)  |
| :-----------: | :---------------------: |
|   **优化后**   | ![9dhnm-lzan3](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/9dhnm-lzan3.gif) ![pn0js-fwsrz](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/pn0js-fwsrz.gif)|

### 2.2 案例2

> 视频中由于镜头切换导致人物的变化幅度过大

|    **优化前**    |![wi9u6-03g86](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/wi9u6-03g86.gif)|
|:-----------:|:-----------:|
|    **优化后**    |![jti4r-arlux](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/jti4r-arlux.gif)![9dvb0-cy6kd](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/9dvb0-cy6kd.gif)![dnutf-d3ko6](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/dnutf-d3ko6.gif)![hb0j2-udinu](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/hb0j2-udinu.gif)![c0xz2-ynmc0](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/c0xz2-ynmc0.gif)![8ycs6-c8otf](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/8ycs6-c8otf.gif)![5a6tv-czm4i](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/5a6tv-czm4i.gif)|

















