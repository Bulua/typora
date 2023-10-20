[toc]



# 支持向量机

## 1、必备知识

### 1.1 点到平面的距离

**如果你已经具备这方面的知识，可以直接跳过....**

![20231017103103](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231017103103.png)

设平面$\prod$的方程为:
$$
\prod : Ax+By+Cz+D=0 \tag{1.1}
$$
设向量$\vec{n}=(A,B,C)$为$\prod$的法向量，平面外一点$M_1$坐标为$(x_1,y_1,z_1)$，在平面上取一点$M_0$，坐标为$(x_0,y_0,z_0)$，则点$M_1$到平面$\prod$的距离$d$为：
$$
d = ||\overrightarrow{M_0 M_1}||cos \alpha	\tag{1.2}
$$
其中，$\alpha$为向量$\vec{n}$与向量$\overrightarrow{M_0 M_1}$之间的夹角，
$$
cos \alpha = \frac{\overrightarrow{M_0 M_1} · \overrightarrow{n}}{\lVert\overrightarrow {M_0M_1}\rVert·\lVert \overrightarrow{n} \rVert}	\tag{1.3}
$$

故：
$$
d = \frac{\overrightarrow{M_0M_1} · \overrightarrow{n}}{\lVert \overrightarrow{n} \rVert} 	\tag{1.4}
$$
而：
$$
\overrightarrow{M_0M_1} · \overrightarrow{n} = |A(x_1-x_0)+B(y_1-y_0)+C(z_1-z_0)|	\tag{1.5}
$$
由于点$M_0$在平面$\prod$上面，所以有：
$$
Ax_0 + By_0 + Cz_0 + D = 0 		\tag{1.6}
$$
可得：
$$
tips: (结合式子1.5和1.6)	\\
\overrightarrow{M_0M_1} · \overrightarrow{n} = |Ax_1 + By_1 + Cz_1 + D|	\tag{1.7}
$$
所以：
$$
tips: (结合式子1.4 和 1.7)	\\
d = \frac{|Ax_1 + By_1 + Cz_1 + D|}{\sqrt{A^2+B^2+C^2}}	\tag{1.8}
$$


# 支持向量机推导流程

## 1、问题定义

归结成一句话：==最大化距离超平面最近点（支持向量）==到该超平面的距离。

<img src="https://raw.githubusercontent.com/Bulua/BlogImageBed/master/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20231019161437.png" alt="1697547945337" style="zoom:50%;" />

这里面超平面我们用$\prod=w_0x_0+w_1x_1+...+w_nx_n$来表示，采用向量的形式来表示为$\prod=w^Tx$，其中$w^T=(w_0,w_1,...,w_n)$，$x=(x_0,x_1,...,x_n)$，用公式来表达上面的问题就是：
$$
max_{w,b}(min_{x_i} \frac{y_i(w^Tx_i+b)}{||w||})	\tag{1.1}
$$

## 2、距离定义

### 2.1 函数距离

当$(w^T, b)$成倍的改变时，函数距离会随着改变，公式如下：
$$
\hat{r_i} = y_i(w^Tx_i+b)	\tag{2.1}
$$

### 2.2 几何距离

当$(w^T, b)$成倍的改变时，几何距离不会受到影响，公式如下：
$$
\frac{\hat{r_i}}{||w||}	\tag{2.2}
$$

## 3、问题转化

### 3.1 最大化几何间隔

我们将问题转化为最大化几何间隔问题，将最近点到超平面的函数距离$\hat{r}=1$，最大化$\frac{1}{||w||}$变为最小化$||w||$，将$||w||$加平方并乘以$\frac{1}{2}$是为了求导的时候好求，不影响最后的结果。
$$
\begin{aligned}
max_{w,b} \frac{\hat{r}}{||w||} \ \ \ \ \ \ \ &\Rightarrow \ \ \ \ \ \ min_{w,b} \frac{1}{2}||w||^2	\\
s.t \ \ y_i(w^Tx_i+b) \geq1 & \ \ \ \ \ \ \ \ s.t \ \ y_i(w^Tx_i+b) \geq1
\end{aligned}
$$

### 3.2 对偶问题

$$
L(w,b,\alpha) = \frac{1}{2}||w||^2-\sum_{i=1}^{n}\alpha_iy_i(wx_i+b)+\sum_{i=1}^{n}\alpha_i	\tag{3.1}
$$

### 3.3 极大极小问题：

在章节4中我们先求$\min_{w,b}$，随后求$\max_{\alpha}$，有了下面的公式：
$$
\max_{\alpha}\min_{w,b}L(w,b,\alpha)		\tag{3.2}
$$

## 4、推导

### 4.1 求极小

$$
\begin{aligned}
\nabla_w L : \ \ \ w - \sum_{i=1}^{n}\alpha_iy_ix_i &= 0	\\
\nabla_b L : \ \ \ \ \ \ \ \ \ \ \ \ \ \ \sum_{i=1}^{n}\alpha_iy_i &= 0
\end{aligned}
$$

### 4.2 代回

将以上公式代回公式3.1可得：
$$
\begin{aligned}
L(w,b,\alpha) &= \frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i^2 \alpha_j^2 y_i y_j(x_ix_j) - \sum_{i=1}^{n}\alpha_i y_i(\sum_{i=1}^{n}\alpha_iy_ix_i^2+b) + \sum_{i=1}^{n}\alpha_i	\\
&= -\frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i^2 \alpha_j^2 y_i y_j(x_ix_j) + \sum_{i=1}^{n}\alpha_i
\end{aligned}
$$

### 4.3 求极大

$$
\max_\alpha \ \ \sum_{i=1}^n \alpha_i -\frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i^2 \alpha_j^2 y_i y_j(x_ix_j) \\
s.t. \ \ \alpha_i \geq 0	\\
\sum_{i=1}^n \alpha_iy_i = 0
$$

## 5、求解

因为$x_i, y_i$是已知的，首先根据4.3中公式求出一系列$\alpha_i$，根据4.1小结的公式，在求出$w$。
$$
w = \sum_{i=1}^{n}\alpha_iy_ix_i	\tag{5.1}
$$
之后，求出$b$即可。
$$
b = \frac{1}{m} \sum_{i=1}^n (y_i - \sum_{j=1}^n x_ix_jy_j\alpha_j)
$$























