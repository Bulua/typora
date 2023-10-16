# 镜头切换检测进展

## 问题描述

由于前后帧可能出现镜头远近切换的情况，导致切片视频包含人物不连续动作的片段。**根据前后帧检测出人物的关键点距离的差异来判断是否出现镜头切换**。

### 1、实现流程

1. 获取（前后帧置信度 > 置信度阈值）的集合$k_1$和$k_2$

2. 取集合$k_1$、$k_2$的交集$intersect$。

3. 获取前后帧在交集中的关键点位置$xy1、xy2$, 并添加坐标原点。

4. 计算==前后帧关键点之间的距离==（用于判断前后帧人物大小是否相似）、==各个关键点和坐标原点的距离==（用于判断前后帧人物位置是否有过大的偏移）$dis_1、dis_2$。

5. 求出$dis_1、dis_2$的距离之和$s_1、s_2$。

6. 用以下公式来计算前后帧人物之间的距离差异：
   $$
   x = \frac{s_1}{s_2} - 1 \\
   Diff(x) = diff = tanh(x) = \frac{e^{2x}-1}{e^{2x}+1}
   $$
   
   以下是`tanh`函数的曲线区间：
   
   <img src="https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231012143832.png" alt="20231012143832" style="zoom: 80%;" />

7. 当前后帧存在镜头切换，差异`diff`过大时，可能存在镜头切换，否则不存在。
8. 描述差异是否过大是通过设计一个阈值`th`来判定，目前设定`th=0.15`。

## 2、优化效果

### 2.1 案例1

> 视频中由于镜头切换导致人物位置变化较大，因此需要切开。

|   **优化前**   | ![otetj-gi6fv](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/otetj-gi6fv.gif)  |
| :-----------: | :---------------------: |
|   **优化后**   | ![9dhnm-lzan3](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/9dhnm-lzan3.gif) ![pn0js-fwsrz](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/pn0js-fwsrz.gif)|

### 2.2 案例2

> 视频中由于镜头切换导致人物的变化幅度过大，因此需要切开

|    **优化前**    |![wi9u6-03g86](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/wi9u6-03g86.gif)|
|:-----------:|:-----------:|
|    **优化后**    |![jti4r-arlux](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/jti4r-arlux.gif)![9dvb0-cy6kd](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/9dvb0-cy6kd.gif)![dnutf-d3ko6](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/dnutf-d3ko6.gif)![hb0j2-udinu](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/hb0j2-udinu.gif)<br>![c0xz2-ynmc0](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/c0xz2-ynmc0.gif)![8ycs6-c8otf](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/8ycs6-c8otf.gif)![5a6tv-czm4i](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/5a6tv-czm4i.gif)|

















