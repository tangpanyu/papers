# Gated DeltaNet Kernel 简化版

这份简化版只保留实现 chunkwise Gated DeltaNet kernel 需要的主线：

```text
单步语义 -> chunkwise 公式 -> W/U_g/D 的区别 -> 一个 C=2,d=3 数值例子 -> kernel checklist
```

本文沿用当前代码里的 state 方向：

```text
S: [dk, dv]
Q,K: [C, dk]
V,D,O: [C, dv]
o_t = q_t^T S_t
```

## 1. 单步语义

Gated DeltaNet 的单步更新可以理解成：

```text
先 decay 旧 state
再用 delta rule 擦除旧内容并写入新 value
```

公式是：

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
o_t=q_t^\top S_t
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

```text
state decay:  alpha_t S_{t-1}
erase:       -beta_t k_t k_t^T alpha_t S_{t-1}
write:       beta_t k_t v_t^T
```

## 2. Chunk 内 decay

对一个 chunk，定义 cumulative decay：

$$
\gamma_r=\prod_{j=1}^{r}\alpha_j
$$

并约定：

$$
\gamma_0=1
$$

chunk 内第 $i$ 个 token 的写入传到第 $r$ 个 token，需要相对衰减：

$$
\Gamma_{r,i}=
\begin{cases}
\gamma_r/\gamma_i,& r\ge i\\
0,& r<i
\end{cases}
$$

所以：

```text
gamma_r:      从 chunk 开头到 token r 的 absolute decay
Gamma_{r,i}:  从 token i 的写入到 token r 的 relative decay
```

## 3. Chunkwise 主公式

chunkwise 计算可以写成：

$$
D=U_g-\overleftarrow W S_0
$$

$$
O=\overleftarrow Q S_0 + ((QK^\top)\odot\Gamma)D
$$

$$
S_C=\gamma_C S_0+\overrightarrow K^\top D
$$

其中：

$$
\overleftarrow Q_r=\gamma_r Q_r
$$

$$
\overleftarrow W_r=\gamma_r W_r
$$

$$
\overrightarrow K_i=\frac{\gamma_C}{\gamma_i}K_i
$$

shape 是：

```text
D:                    [C, dv]
Q_left @ S0:           [C, dk] @ [dk, dv] -> [C, dv]
((QK^T) * Gamma) @ D:  [C, C]  @ [C, dv]  -> [C, dv]
K_right.T @ D:         [dk, C] @ [C, dv]  -> [dk, dv]
```

## 4. W 和 U_g 为什么不一样

普通 DeltaNet 里，chunk 内依赖由 lower-triangular solve 吸收：

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

Gated DeltaNet 多了 decay，所以写入侧变成 decay-aware 的 $U_g$：

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

关键区别：

```text
U_g:
  处理 chunk 内第 i 个写入影响第 r 个 token
  这条路径是 i -> r
  所以 T/A_u 里要乘 Gamma_{r,i} = gamma_r / gamma_i

W:
  处理 chunk 开头传入的旧 state S0
  S0 等价于来自虚拟位置 0
  这条路径是 0 -> r
  所以衰减是 gamma_r / gamma_0 = gamma_r
  这个 gamma_r 只和当前行有关，可以最后按行乘到 W 上
```

因此：

$$
\overleftarrow W_r=\gamma_r W_r
$$

而不是在 $W$ 的 lower-triangular solve 里再乘 $\Gamma$。

代码心智模型：

```text
u / k_cumsum:
  U_g = A_u @ (beta * V)
  A_u 的 strict-lower 里带 Gamma mask

w / k_cumdecay:
  W = A_w @ (beta * K)
  A_w 的 strict-lower 里不带 Gamma mask

chunk_fwd_h_fn:
  W_left = gamma * W
  D = U_g - W_left @ S
```

## 5. C=2,d=3 数值例子

取：

$$
S_0=I
$$

$$
\alpha_1=0.5,\quad \beta_1=0.8
$$

$$
\alpha_2=0.25,\quad \beta_2=0.5
$$

$$
Q=
\begin{bmatrix}
1 & 1 & 0.5\\
1 & -1 & 2
\end{bmatrix}
$$

$$
K=
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$$

$$
V=
\begin{bmatrix}
2 & 1 & 0.5\\
1 & 3 & -1
\end{bmatrix}
$$

这里：

$$
\gamma_1=0.5,\quad \gamma_2=0.125
$$

$$
\Gamma=
\begin{bmatrix}
1 & 0\\
0.25 & 1
\end{bmatrix}
$$

因为 $k_1,k_2$ 正交，所以 $A_w,A_u$ 都退化成单位阵：

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

擦旧 state 时按行乘 $\gamma$：

$$
\overleftarrow W=
\begin{bmatrix}
0.4 & 0 & 0\\
0 & 0.0625 & 0
\end{bmatrix}
$$

因此：

$$
D=U_g-\overleftarrow W S_0
$$

$$
D=
\begin{bmatrix}
1.2 & 0.8 & 0.4\\
0.5 & 1.4375 & -0.5
\end{bmatrix}
$$

旧 state 读取：

$$
\overleftarrow Q=
\begin{bmatrix}
0.5 & 0.5 & 0.25\\
0.125 & -0.125 & 0.25
\end{bmatrix}
$$

chunk 内读取：

$$
(QK^\top)\odot\Gamma=
\begin{bmatrix}
1 & 0\\
0.25 & -1
\end{bmatrix}
$$

最终输出：

$$
O=
\overleftarrow Q S_0
+
((QK^\top)\odot\Gamma)D
$$

$$
O=
\begin{bmatrix}
1.7 & 1.3 & 0.65\\
-0.075 & -1.3625 & 0.85
\end{bmatrix}
$$

state update 用：

$$
\overrightarrow K=
\begin{bmatrix}
0.25 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$$

$$
S_{\text{next}}
=
\gamma_2S_0+\overrightarrow K^\top D
$$

$$
S_{\text{next}}=
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
1. 读 Q,K,V,beta,alpha
2. 计算 gamma = cumulative_prod(alpha)
3. 构造 Gamma_{r,i} = gamma_r / gamma_i, r >= i
4. 计算 KKT = K @ K.T
5. 计算 A_u = inverse_lower(I + strict_lower(diag(beta) @ (Gamma * KKT)))
6. 计算 U_g = A_u @ (beta * V)
7. 计算 A_w = inverse_lower(I + strict_lower(diag(beta) @ KKT))
8. 计算 W = A_w @ (beta * K)
9. 计算 Q_left = gamma * Q
10. 计算 W_left = gamma * W
11. 计算 D = U_g - W_left @ S
12. 计算 O = Q_left @ S + ((Q @ K.T) * Gamma) @ D
13. 计算 K_right = (gamma_C / gamma) * K
14. 计算 S_next = gamma_C * S + K_right.T @ D
```

实现注意：

```text
T/A_u/A_w 是 C x C lower triangular，小 chunk 可以放 shared memory。
inverse_lower 不是真的求通用逆，而是 unit lower-triangular forward substitution。
输出前如果代码有 q scale，要和 reference 保持一致。
当前例子中 S0=I 只是为了手算清楚，实际代码常从零 state 开始。
```

## 7. 和 KDA 的关系

KDA 可以看成把 Gated DeltaNet 的 scalar decay 推广成 channel-wise decay。

GDN 里：

$$
\gamma_r/\gamma_i
$$

是一个 scalar。

KDA 里通常用 log cumulative decay：

$$
g_r=\sum_{j=1}^{r}\log\alpha_j
$$

于是相对衰减写成：

$$
\exp(g_r-g_i)
$$

如果是 channel-wise decay，这个量就是一个向量，需要按 key channel 缩放 $K$，不能再简单当成一个 $C\times C$ scalar mask。
