[toc]

# 逻辑回归的推导流程

## 1、LR推导
$$
P(Y=1|x) = \frac{e^{wx}}{e^{wx}+1} = \pi(x) \\ \tag{1.1}
$$

$$
P(Y=0|x) = \frac{1}{e^{wx}+1} = 1 - \pi(x)  \\ \tag{1.2}
$$



![微信截图_20231016205213](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20231016205213.png)

## 2、似然函数

$$
L(w) = \prod_{i=1}^N \ [\pi(x_i)]^{y_i} \ [\pi(1-x_i)]^{1-y_i}  \tag{2.1}
$$

## 3、对数似然

$$
L(w) &= \sum_{i=1}^{N} \ [y_i · log \pi(x_i) \ + \ (1-y_i)log(1-\pi(x_i))]  \\
&= \sum_{i=1}^{N} \ [y_i·{wx_i} \ - \ log(e^{wx_i}+1)] \\	\tag{3.1}
$$

具体流程如下：
$$
\begin{aligned}
L(w) &= \sum_{i=1}^{N} \ [y_i · log \pi(x_i) \ + \ (1-y_i)log(1-\pi(x_i))]  \\
&= \sum_{i=1}^{N} \ [y_i · log \pi(x_i) \ - \ y_i log(1-\pi(x_i)) \ + \ log(1-\pi(x_i))]  \\
&= \sum_{i=1}^{N} \ [y_i · log \frac{\pi(x_i)}{1-\pi(x_i)} \ + \ log(1-\pi(x_i))]   \\
(tips: &由式子1.1可得出e^{wx} = \frac{\pi(x)}{1-\pi(x)})    \\
&= \sum_{i=1}^{N} \ [y_i·log e^{wx_i} + log\frac{1}{e^{wx_i}+1}]  \\
&= \sum_{i=1}^{N} \ [y_i·{wx_i} \ - \ log(e^{wx_i}+1)] \\
\end{aligned}
$$

## 4、求导

$$
\begin{aligned}
\frac{\partial L(w)}{\partial w} &= \sum_{i=1}^{N} \ [y_ix_i \ - \ \frac{e^{wx_i}}{e^{wx_i}+1}x_i] \\
(tips: 由式子&1.1可得出\frac{e^{wx_i}}{e^{wx_i}+1} = \pi(x_i))    \\
&= \sum_{i=1}^{N} \ [y_ix_i \ - \ \pi(x_i)x_i]
\end{aligned}   \tag{4.1}
$$

## 5、更新

$$
\begin{aligned}
w &= w_0 + \alpha\frac{\partial L(w)}{\partial w}	\\
&= w_0 + \alpha · x_i(y_i \ - \ \pi(x_i))   
\end{aligned}
$$