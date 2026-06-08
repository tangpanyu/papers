# KDA Kernel 简化版

这份笔记按 `Gated DeltaNet Kernel 简化版.md` 的顺序写，方便逐节对照实现：

```text
单步语义 -> chunk 内 decay -> chunkwise 主公式 -> W/U/D 的区别 -> C=2,d=3 数值例子 -> kernel checklist
```

先记住一句话：

```text
KDA 用固定大小的 state S 代替完整 KV cache。
它和 GDN 的主线一样：decay 旧 state，再用 delta rule 擦写。
区别是：GDN 的 decay 是 scalar，KDA 的 decay 是 key/channel-wise vector。
```

本文省略 batch/head 维度，只看一个 chunk。令

$$
Q,K,\alpha,g,\gamma,W,K_{\text{right}}\in\mathbb{R}^{C\times d_k}
$$

$$
V,U,D,O\in\mathbb{R}^{C\times d_v}
$$

$$
S_0,S_C\in\mathbb{R}^{d_k\times d_v}
$$

$$
\beta\in\mathbb{R}^{C},\qquad
B,E,M\in\mathbb{R}^{C\times C}
$$

代码里的完整张量只是多了 batch/head 维：

$$
q,k,g,\gamma\in\mathbb{R}^{B\times H\times T\times d_k},\quad
v,o\in\mathbb{R}^{B\times H\times T\times d_v},\quad
S\in\mathbb{R}^{B\times H\times d_k\times d_v}
$$

KDA kernel 的入口是：

```text
输入: Q, K, V, alpha/log_alpha, beta, initial_state
输出: O, final_state
```

decode 时一次只处理一个 token，直接用 recurrent update。prefill/training 时一次处理一段 prompt，所以要按 chunk 并行。

论文 Appendix C 的伪代码里还会做：

$$
q\leftarrow q\cdot d_k^{-1/2}
$$

本文后面的公式默认 $q$ 已经完成这个 scale；写 reference 时要和这个约定保持一致。

state 方向固定为：

$$
S_t\in\mathbb{R}^{d_k\times d_v},\qquad
o_t=S_t^\top q_t\in\mathbb{R}^{d_v}
$$

## 1. 单步语义

对一个 token，

$$
q_t,k_t,\alpha_t\in\mathbb{R}^{d_k},\quad
v_t,d_t,o_t\in\mathbb{R}^{d_v},\quad
\beta_t\in\mathbb{R}
$$

KDA 的单步公式是论文 Eq. 1：

$$
S_t
=
\left(I-\beta_t k_tk_t^\top\right)\operatorname{Diag}(\alpha_t)S_{t-1}
+
\beta_tk_tv_t^\top
$$

和 GDN 一样，把它拆成“先 decay，再 delta 擦写”更好懂：

$$
\bar S_t=\operatorname{Diag}(\alpha_t)S_{t-1}
$$

这个 $\bar S_t$ 只是把原公式里的 decayed old state 单独命名。把它代回原式：

$$
S_t
=
\left(I-\beta_t k_tk_t^\top\right)\bar S_t
+
\beta_tk_tv_t^\top
$$

展开括号：

$$
S_t
=
\bar S_t
-
\beta_t k_tk_t^\top\bar S_t
+
\beta_tk_tv_t^\top
$$

其中：

$$
k_t^\top\bar S_t
=
(\bar S_t^\top k_t)^\top
$$

所以定义：

$$
\hat v_t=\bar S_t^\top k_t
$$

它表示“decayed old state 在当前 key $k_t$ 上已经读出来的 old value”。于是擦除项可以写成：

$$
\beta_t k_tk_t^\top\bar S_t
=
\beta_t k_t\hat v_t^\top
$$

再和写入项合并：

$$
S_t
=
\bar S_t
+
\beta_tk_t(v_t-\hat v_t)^\top
$$

因此定义 correction：

$$
d_t=\beta_t(v_t-\hat v_t)
$$

就得到：

$$
S_t=\bar S_t+k_td_t^\top
$$

$$
o_t=S_t^\top q_t
$$

最小 decode reference：

```python
# q, k, alpha: [dk]
# v:           [dv]
# beta:        scalar
# S:           [dk, dv]

S_decay = alpha[:, None] * S
old = S_decay.T @ k
d = beta * (v - old)
S = S_decay + k[:, None] @ d[None, :]
o = S.T @ q
```

这里最重要的区别是：

$$
\text{GDN:}\quad \bar S_t=\alpha_tS_{t-1}
$$

$$
\text{KDA:}\quad \bar S_t=\operatorname{Diag}(\alpha_t)S_{t-1}
$$

GDN 是整张 state 共享一个衰减；KDA 是 state 的每个 key/channel 行有自己的衰减。

## 2. Chunk 内 decay

GDN 里 $\alpha_t$ 是 scalar，所以 $\gamma_r$ 也是 scalar。KDA 里 $\alpha_t$ 是 $d_k$ 维向量，所以累计 decay 也是 $d_k$ 维向量。

论文 Appendix C 通常输入 log decay。进入一个 chunk 后先做：

```python
g = g.cumsum(token_dim)
```

本文记：

$$
g^{raw}_r=\log\alpha_r
$$

$$
g_r=\sum_{j=1}^{r}g^{raw}_j
$$

$$
\gamma_r=\exp(g_r)=\prod_{j=1}^{r}\alpha_j
$$

其中：

$$
g_r,\gamma_r\in\mathbb{R}^{d_k}
$$

第 $i$ 个 token 的写入传到第 $r$ 个 token，需要 channel-wise 相对衰减：

$$
\rho_{r,i}=\exp(g_r-g_i)=\gamma_r/\gamma_i\in\mathbb{R}^{d_k}
$$

直觉：

$$
\gamma_r:\text{ chunk 起点到 }r\text{ 的每个 channel 的 decay}
$$

$$
\rho_{r,i}:\text{ token }i\text{ 的写入传到 }r\text{ 时，每个 channel 还剩多少}
$$

如果实现输入的是 $\alpha$，先取 $\log\alpha$ 得到 $g^{raw}$；如果实现已经输入 log gate，就直接做 cumsum。

## 3. Chunkwise 主公式

先定义三组 decay 后的矩阵：

$$
Q_{\text{abs}}=\gamma\odot Q
$$

$$
K_{\text{abs}}=\gamma\odot K
$$

$$
K_{\text{right},i}=\rho_{C,i}\odot k_i=\exp(g_C-g_i)\odot k_i
$$

KDA 的 chunk 内 query-key score 不是 GDN 的：

$$
(QK^\top)\odot\Gamma
$$

而是：

$$
E=\operatorname{Tril}\left(Q_{\text{abs}}\left(\frac{K}{\gamma}\right)^\top\right)
$$

也就是逐元素写成：

$$
E_{r,i}=
\begin{cases}
q_r^\top(\rho_{r,i}\odot k_i),& r\ge i\\
0,& r<i
\end{cases}
$$

chunkwise 主公式是：

$$
\boxed{
D=U-WS_0
}
$$

$$
\boxed{
O=Q_{\text{abs}}S_0+ED
}
$$

$$
\boxed{
S_C=\operatorname{Diag}(\gamma_C)S_0+K_{\text{right}}^\top D
}
$$

这三行和 GDN 简化版一一对应：

```text
D:   先算真正要写入/读取的 pseudo-value
O:   旧 state 贡献 + chunk 内新写入贡献
S_C: decay 旧 state + 写入当前 chunk
```

注意，$K/\gamma$ 是数学写法。真正实现时通常不显式 materialize 这个除法，而是在构造每个 score 时用：

$$
\exp(g_r-g_i)
$$

## 4. W、U、D 为什么这样定义

单步里：

$$
d_r=\beta_r(v_r-\bar S_r^\top k_r)
$$

但 $\bar S_r$ 已经包含前面 token 的写入，所以 $d_r$ 依赖 $d_1,\dots,d_{r-1}$。

把前面的展开代进去，可以得到：

$$
d_r
=
\beta_r v_r
-
\beta_r S_0^\top(\gamma_r\odot k_r)
-
\beta_r
\sum_{i<r}
d_i\,k_r^\top(\rho_{r,i}\odot k_i)
$$

这里出现 key-key interaction：

$$
B_{r,i}=k_r^\top(\rho_{r,i}\odot k_i)
$$

矩阵写法对应论文 Eq. 6：

$$
B=(\gamma\odot K)\left(\frac{K}{\gamma}\right)^\top
$$

然后用 lower-triangular solve 吸收 chunk 内依赖：

$$
M=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
\right]^{-1}
\operatorname{Diag}(\beta)
$$

这里的 inverse 不是通用矩阵求逆，而是 unit lower-triangular forward substitution。

论文 Eq. 7 定义：

$$
W=M(\gamma\odot K)
$$

$$
U=MV
$$

$$
D=U-WS_0
$$

直觉：

```text
U:  只看当前 chunk 的 value，会写进去什么
W:  当前 chunk 会从旧 state 里擦掉/扣掉什么
D:  扣掉旧 state 影响后的有效写入 pseudo-value
```

为什么 $D$ 要减 $WS_0$？因为 delta rule 不是盲写 $v_r$，而是写：

$$
v_r-\text{old value}
$$

旧 value 来自 chunk 开始前的 state $S_0$，所以要从 $U$ 里扣掉 $WS_0$。

实现里推荐按 Appendix C 的伪代码直接算：

$$
B_{r,i}=\operatorname{dot}(k_r\odot\exp(g_r-g_i),k_i)
$$

$$
E_{r,i}=\operatorname{dot}(q_r\odot\exp(g_r-g_i),k_i)
$$

## 5. C=2,d=3 数值例子

取：

$$
C=2,\qquad d_k=d_v=3,\qquad S_0=I_3
$$

沿用 GDN 例子里的 $Q,K,V,\beta$：

$$
\beta_1=0.8,\quad \beta_2=0.5
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

KDA 的 channel-wise decay 取：

$$
\alpha_1=
\begin{bmatrix}
0.5 & 0.8 & 1
\end{bmatrix}
$$

$$
\alpha_2=
\begin{bmatrix}
0.25 & 0.5 & 0.75
\end{bmatrix}
$$

所以：

$$
\gamma_1=
\begin{bmatrix}
0.5 & 0.8 & 1
\end{bmatrix},
\qquad
\gamma_2=
\begin{bmatrix}
0.125 & 0.4 & 0.75
\end{bmatrix}
$$

因为 $k_1\perp k_2$，所以 $M=\operatorname{Diag}(\beta)$。absolute-decayed key 是：

$$
K_{\text{abs}}=
\gamma\odot K
=
\begin{bmatrix}
0.5 & 0 & 0\\
0 & 0.4 & 0
\end{bmatrix}
$$

于是：

$$
W=M K_{\text{abs}}
=
\begin{bmatrix}
0.4 & 0 & 0\\
0 & 0.2 & 0
\end{bmatrix}
$$

$$
U=MV
=
\begin{bmatrix}
1.6 & 0.8 & 0.4\\
0.5 & 1.5 & -0.5
\end{bmatrix}
$$

$$
D=U-WS_0
=
\begin{bmatrix}
1.2 & 0.8 & 0.4\\
0.5 & 1.3 & -0.5
\end{bmatrix}
$$

旧 state 读取：

$$
Q_{\text{abs}}=
\gamma\odot Q
=
\begin{bmatrix}
0.5 & 0.8 & 0.5\\
0.125 & -0.4 & 1.5
\end{bmatrix}
$$

chunk 内读取矩阵：

$$
E=
\begin{bmatrix}
1 & 0\\
0.25 & -1
\end{bmatrix}
$$

输出：

$$
O=
Q_{\text{abs}}S_0+ED
=
\begin{bmatrix}
1.7 & 1.6 & 0.9\\
-0.075 & -1.5 & 2.1
\end{bmatrix}
$$

state update 需要：

$$
K_{\text{right}}=
\begin{bmatrix}
0.25 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$$

因此：

$$
S_{\text{next}}
=
\operatorname{Diag}(\gamma_2)S_0+K_{\text{right}}^\top D
=
\begin{bmatrix}
0.425 & 0.2 & 0.1\\
0.5 & 1.7 & -0.5\\
0 & 0 & 0.75
\end{bmatrix}
$$

这和逐 token recurrent reference 完全一致。

## 6. Kernel Checklist

最小实现流程：

```text
1. g = cumsum(g_raw, token_dim)
2. gamma = exp(g)
3. B[r,i] = dot(k[r] * exp(g[r]-g[i]), k[i])
4. M = inverse_lower(I + strict_lower(diag(beta) @ B)) @ diag(beta)
5. W = M @ (gamma * K)
6. U = M @ V
7. D = U - W @ S
8. E[r,i] = dot(q[r] * exp(g[r]-g[i]), k[i]), i <= r
9. O = (gamma * Q) @ S + E @ D
10. K_right = exp(g_C - g) * K
11. S_next = gamma_C[:, None] * S + K_right.T @ D
```

实现注意：

```text
q 要和论文伪代码一样乘 d_k^{-0.5}，或者和 reference 保持同一约定。
inverse_lower 不是真的求通用逆，而是 unit lower-triangular forward substitution。
K/gamma 只是数学写法，实际优先用 exp(g_r-g_i)。
Q_abs、K_abs、K_right 都可以不 materialize，在 load 时乘 decay。
```

和论文对应：

```text
Eq. 1: 单步 recurrent
Eq. 6: B 和 M 的 lower-triangular solve
Eq. 7: W = M(gamma*K), U = M V
Eq. 8: S_next
Eq. 9: O = old state 贡献 + chunk 内贡献
Appendix C Listing 1: g.cumsum 和 exp(g_r-g_i)
```

## 7. 和 GDN 的关系

KDA 可以看成把 GDN 的 scalar decay 推广成 channel-wise decay。

GDN 里：

$$
\Gamma_{r,i}=\gamma_r/\gamma_i\in\mathbb{R}
$$

所以：

$$
q_r^\top(\Gamma_{r,i}k_i)=\Gamma_{r,i}(q_r^\top k_i)
$$

它可以先算 $QK^\top$，再乘 $C\times C$ 的 scalar mask。

KDA 里：

$$
\rho_{r,i}=\exp(g_r-g_i)\in\mathbb{R}^{d_k}
$$

所以：

$$
q_r^\top(\rho_{r,i}\odot k_i)
=
\sum_c q_{r,c}\rho_{r,i,c}k_{i,c}
$$

因为每个 channel 的 $\rho_{r,i,c}$ 都可能不同，它不能从求和里提出去。只有当所有 channel 的衰减都相同，KDA 才退化成 GDN 那种 scalar mask。

实现开销上，可以这样简化理解：

```text
GDN:
  gamma 是 [C]
  score = (QK.T) * Gamma
  state decay 是整张 state 乘一个数

KDA:
  gamma 是 [C, dk]
  score = dot(q[r] * exp(g[r]-g[i]), k[i])
  state decay 是每一行乘一个数
```

主复杂度没有质变，构造 $B/E$ 的主项仍然是 $O(C^2d_k)$。KDA 多出来的主要是 $O(Cd_k)$ 的 channel-wise scaling、`exp` 和中间值管理；换来的是同一个 head 内可以同时有长记忆 channel 和短记忆 channel。
