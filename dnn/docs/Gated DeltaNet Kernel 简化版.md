# Gated DeltaNet Kernel 简化版

这份笔记只保留实现 chunkwise Gated DeltaNet kernel 需要的主线：

```text
单步语义 -> chunkwise 公式 -> W/U_g/D 的区别 -> C=2,d=3 数值例子 -> kernel checklist
```

本文省略 batch/head 维度，只看一个 chunk。令

$$
Q,K,W,\overleftarrow W,\overrightarrow K\in\mathbb{R}^{C\times d_k}
$$

$$
V,U_g,D,O\in\mathbb{R}^{C\times d_v}
$$

$$
S_0,S_C\in\mathbb{R}^{d_k\times d_v}
$$

$$
\alpha,\beta,\gamma\in\mathbb{R}^{C},\qquad
\Gamma,A_u,A_w\in\mathbb{R}^{C\times C}
$$

代码里的完整张量只是多了 batch/head 维：

$$
q,k\in\mathbb{R}^{B\times H\times T\times d_k},\quad
v,o\in\mathbb{R}^{B\times H\times T\times d_v},\quad
S\in\mathbb{R}^{B\times H\times d_k\times d_v}
$$

state 方向固定为：

$$
S_t\in\mathbb{R}^{d_k\times d_v},\qquad
o_t=S_t^\top q_t\in\mathbb{R}^{d_v}
$$

## 1. 单步语义

对一个 token，

$$
q_t,k_t\in\mathbb{R}^{d_k},\quad
v_t,d_t,o_t\in\mathbb{R}^{d_v},\quad
\alpha_t,\beta_t\in\mathbb{R}
$$

Gated DeltaNet 可以理解成先 decay 旧 state，再用 delta rule 擦写：

$$
\bar S_t=\alpha_t S_{t-1}
$$

$$
d_t=\beta_t(v_t-\bar S_t^\top k_t)
$$

$$
S_t=\bar S_t+k_td_t^\top
$$

$$
o_t=S_t^\top q_t
$$

等价展开：

$$
S_t
=
\alpha_tS_{t-1}
-\beta_tk_tk_t^\top(\alpha_tS_{t-1})
+\beta_tk_tv_t^\top
$$

也就是：

$$
\underbrace{\alpha_tS_{t-1}}_{\text{decay}}
\quad
\underbrace{-\beta_tk_tk_t^\top(\alpha_tS_{t-1})}_{\text{erase}}
\quad
\underbrace{+\beta_tk_tv_t^\top}_{\text{write}}
$$

## 2. Chunk 内 decay

对一个长度为 $C$ 的 chunk，定义 cumulative decay：

$$
\gamma_0=1,\qquad
\gamma_r=\prod_{j=1}^{r}\alpha_j
$$

第 $i$ 个 token 的写入传到第 $r$ 个 token 时，需要相对衰减：

$$
\Gamma_{r,i}=
\begin{cases}
\gamma_r/\gamma_i,& r\ge i\\
0,& r<i
\end{cases}
$$

直觉：

$$
\gamma_r:\text{ chunk 起点到 }r\text{ 的 decay}
$$

$$
\Gamma_{r,i}:\text{ token }i\text{ 的写入传到 }r\text{ 的 decay}
$$

## 3. Chunkwise 主公式

先定义三组 decay 后的矩阵：

$$
\overleftarrow Q=\operatorname{diag}(\gamma)Q
$$

$$
\overleftarrow W=\operatorname{diag}(\gamma)W
$$

$$
\overrightarrow K=\operatorname{diag}(\gamma_C/\gamma)K
$$

这里 $\gamma_C/\gamma$ 表示逐元素除法：

$$
(\gamma_C/\gamma)_i=\gamma_C/\gamma_i
$$

chunkwise 公式是：

$$
\boxed{
D=U_g-\overleftarrow W S_0
}
$$

$$
\boxed{
O=\overleftarrow Q S_0+\left((QK^\top)\odot\Gamma\right)D
}
$$

$$
\boxed{
S_C=\gamma_C S_0+\overrightarrow K^\top D
}
$$

如果只看维度，三个式子分别是：

$$
\mathbb{R}^{C\times d_v}
=
\mathbb{R}^{C\times d_v}
-
\mathbb{R}^{C\times d_k}
\mathbb{R}^{d_k\times d_v}
$$

$$
\mathbb{R}^{C\times d_v}
=
\mathbb{R}^{C\times d_k}
\mathbb{R}^{d_k\times d_v}
+
\mathbb{R}^{C\times C}
\mathbb{R}^{C\times d_v}
$$

$$
\mathbb{R}^{d_k\times d_v}
=
\mathbb{R}^{d_k\times d_v}
+
\mathbb{R}^{d_k\times C}
\mathbb{R}^{C\times d_v}
$$

这三行就是 kernel 的骨架：

```text
D:   先算真正要写入/读取的 pseudo-value
O:   旧 state 贡献 + chunk 内新写入贡献
S_C: decay 旧 state + 写入当前 chunk
```

## 4. W 和 U_g 为什么不一样

普通 DeltaNet 里，chunk 内依赖由一个 lower-triangular solve 吸收：

$$
A_w=
\left[
I+\operatorname{strictLower}\left(
\operatorname{diag}(\beta)KK^\top
\right)
\right]^{-1}
$$

$$
W=A_w\operatorname{diag}(\beta)K
$$

Gated DeltaNet 多了 scalar decay。写入侧要知道“第 $i$ 个写入传到第 $r$ 个位置时已经衰减多少”，所以用 decay-aware 的：

$$
A_u=
\left[
I+\operatorname{strictLower}\left(
\operatorname{diag}(\beta)(\Gamma\odot KK^\top)
\right)
\right]^{-1}
$$

$$
U_g=A_u\operatorname{diag}(\beta)V
$$

区别可以这样记：

$$
U_g:\quad i\to r,\quad \text{chunk 内写入传播，所以 strict-lower 里带 }\Gamma_{r,i}
$$

$$
W:\quad 0\to r,\quad \text{旧 state 来自 chunk 起点，所以最后按行乘 }\gamma_r
$$

因此擦旧 state 用的是：

$$
\overleftarrow W=\operatorname{diag}(\gamma)W
$$

而不是在 $A_w$ 里再乘 $\Gamma$。

代码心智模型：

```text
U_g = A_u @ (beta * V)          # strict-lower 里带 Gamma
W   = A_w @ (beta * K)          # strict-lower 里不带 Gamma
D   = (U_g) - (gamma * W) @ S
```

## 5. C=2,d=3 数值例子

取：

$$
C=2,\qquad d_k=d_v=3,\qquad S_0=I_3
$$

$$
\alpha_1=0.5,\quad \beta_1=0.8,\qquad
\alpha_2=0.25,\quad \beta_2=0.5
$$

$$
Q=
\begin{bmatrix}
1 & 1 & 0.5\\
1 & -1 & 2
\end{bmatrix},
\quad
K=
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0
\end{bmatrix},
\quad
V=
\begin{bmatrix}
2 & 1 & 0.5\\
1 & 3 & -1
\end{bmatrix}
$$

累计 decay：

$$
\gamma_1=0.5,\qquad \gamma_2=0.125
$$

$$
\Gamma=
\begin{bmatrix}
1 & 0\\
0.25 & 1
\end{bmatrix}
$$

因为 $k_1\perp k_2$，所以 $A_w=A_u=I_2$。于是：

$$
W=
\begin{bmatrix}
0.8 & 0 & 0\\
0 & 0.5 & 0
\end{bmatrix}
$$

$$
U_g=
\begin{bmatrix}
1.6 & 0.8 & 0.4\\
0.5 & 1.5 & -0.5
\end{bmatrix}
$$

擦旧 state 的矩阵：

$$
\overleftarrow W=
\operatorname{diag}(\gamma)W
=
\begin{bmatrix}
0.4 & 0 & 0\\
0 & 0.0625 & 0
\end{bmatrix}
$$

所以：

$$
D=U_g-\overleftarrow W S_0
=
\begin{bmatrix}
1.2 & 0.8 & 0.4\\
0.5 & 1.4375 & -0.5
\end{bmatrix}
$$

旧 state 读取：

$$
\overleftarrow Q=
\operatorname{diag}(\gamma)Q
=
\begin{bmatrix}
0.5 & 0.5 & 0.25\\
0.125 & -0.125 & 0.25
\end{bmatrix}
$$

chunk 内读取矩阵：

$$
(QK^\top)\odot\Gamma=
\begin{bmatrix}
1 & 0\\
0.25 & -1
\end{bmatrix}
$$

输出：

$$
O=
\overleftarrow Q S_0
+
\left((QK^\top)\odot\Gamma\right)D
=
\begin{bmatrix}
1.7 & 1.3 & 0.65\\
-0.075 & -1.3625 & 0.85
\end{bmatrix}
$$

state update 需要：

$$
\overrightarrow K=
\operatorname{diag}(\gamma_2/\gamma)K
=
\begin{bmatrix}
0.25 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$$

因此：

$$
S_{\text{next}}
=
\gamma_2S_0+\overrightarrow K^\top D
=
\begin{bmatrix}
0.425 & 0.2 & 0.1\\
0.5 & 1.5625 & -0.5\\
0 & 0 & 0.125
\end{bmatrix}
$$

这和逐 token recurrent reference 完全一致。

## 6. Kernel Checklist

最小实现流程：

```text
1. gamma = cumulative_prod(alpha)
2. Gamma[r,i] = gamma[r] / gamma[i], i <= r
3. KKT = K @ K.T
4. A_u = inverse_lower(I + strict_lower(diag(beta) @ (Gamma * KKT)))
5. U_g = A_u @ (diag(beta) @ V)
6. A_w = inverse_lower(I + strict_lower(diag(beta) @ KKT))
7. W = A_w @ (diag(beta) @ K)
8. Q_left = diag(gamma) @ Q
9. W_left = diag(gamma) @ W
10. D = U_g - W_left @ S
11. O = Q_left @ S + ((Q @ K.T) * Gamma) @ D
12. K_right = diag(gamma_C / gamma) @ K
13. S_next = gamma_C * S + K_right.T @ D
```

实现注意：

```text
inverse_lower 不是真的求通用逆，而是 unit lower-triangular forward substitution。
输出前如果代码有 q scale，要和 recurrent reference 保持一致。
本文用 S0=I 只是为了手算清楚，实际代码常从零 state 开始。
Q_left、W_left、K_right 都可以不 materialize，在 load 时乘 decay。
```

## 7. 和 KDA 的关系

KDA 可以看成把 Gated DeltaNet 的 scalar decay 推广成 channel-wise decay。

GDN 里：

$$
\gamma_r/\gamma_i\in\mathbb{R}
$$

KDA 里：

$$
\exp(g_r-g_i)\in\mathbb{R}^{d_k}
$$

所以 KDA 的相对 decay 需要按 key channel 缩放 $K$，不能再简单当成一个 $C\times C$ scalar mask。
