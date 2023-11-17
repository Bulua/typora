[toc]

# 卷积神经网络

假设输入`x`的`shape`为($bs, ch_{in}, h_{in}, w_{in}$)，卷积核的数量为$ch_{out}$, 长宽为($k_h, k_w$)，`padding`为($p_h, p_w$)，`stride`为($s_h, s_w$)，则`x`经过该卷积核之后的`shape`($bs, ch_{out}, h_{out}, w_{out}$)公式可用一下来计算：

## 1、输入经过卷积核之后的大小计算

这个比较简单，相信大家都能够推算出来，这里不多叙述。

$$
\begin{aligned}
    h_{out} &= \frac{h_{in} + 2*p_{h} - k_h}{s_h} + 1    \\
    w_{out} &= \frac{w_{in} + 2*p_{w} - k_w}{s_w} + 1
\end{aligned}
$$

## 2、卷积核参数量计算

首先，我们要明白一个卷积核仅有`1`个偏执参数，我们先算一个卷积核有多少参数，每个卷积核的通道是和输入通道$ch_{in}$相等的，也就是每个卷积核有$k_h * k_w * ch_{in}$个参数（不含偏执），一共有$ch_{out}$个卷积核，那么就有$k_h * k_w * ch_{in} * ch_{out}$个参数（不含偏执），最后加上${ch_{out}}$个偏执，就得到了总的参数量$param$。

$$
\begin{aligned}
    param &= k_h * k_w * ch_{out} * ch_{in}  + 1 * ch_{out} \\
    &= (k_h * k_w * ch_{in}  + 1) * ch_{out}
\end{aligned}
$$

## 3、计算量

$FLOPs$ 是`floating point of operations`的缩写，是浮点运算次数，理解为计算量，可以用来衡量算法/模型复杂度。
这里推算我们反着来，从结果看，我们得到了($ch_{out}, h_{out}, w_{out}$)个结果, 我们先计算一个结果的计算量`f`，之后将其×($ch_{out}, h_{out}, w_{out}$)即可得到全部计算量。

要计算`f`，我们知道这个结果是（1）由卷积核扫描到的区域与卷积核相乘，（2）将所有结果相加，（3）最后加上偏执得到的。假设扫描到的区域为$n*n*c$，卷积核大小也应该为$n*n*c$，点积我们可以得到$n*n*c$个数，这时我们的乘法计算量为$n*n*c$, 将这些数相加需要$n*n*c-1$个加法，最后加上偏执，则计算量需要再+1，将这些计算量相加得到一个结果的计算量；共有($ch_{out}, h_{out}, w_{out}$)个结果，最后乘以结果的数量就得到了总的计算量。

$$
\begin{aligned}
    FLOPs &= [(ch_{in} * k_h * k_w) + (ch_{in} * k_h * k_w - 1) + 1] * ch_{out} * h_{out} * w_{out}  \\
    &= 2 * ch_{in} * k_h * k_w * ch_{out} * h_{out} * w_{out}
\end{aligned}
$$
