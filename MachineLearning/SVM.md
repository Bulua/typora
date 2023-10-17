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
























