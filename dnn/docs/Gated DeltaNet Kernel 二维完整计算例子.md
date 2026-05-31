# Gated DeltaNet Kernel：chunk size 2、dim 3 完整计算例子

这份笔记用一个很小但 shape 不会混淆的例子解释 Gated DeltaNet：

$$
C=2,\quad d_k=d_v=3
$$

也就是：

```text
chunk size = 2
q/k/v dim  = 3
Q/K/V      = [2, 3]
S          = [3, 3]
O          = [2, 3]
```

这个例子比 $2\times2$ 好一点：chunk 维度和 feature 维度不一样，不容易把 $C$ 和 $d$ 搞混。

## 0. 问题设置

只看一个 batch、一个 head，省略 $B,H$。

初始状态取单位矩阵：

$$
S_0=
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}
$$

两个 token 的输入：

$$
k_1=
\begin{bmatrix}
1\\0\\0
\end{bmatrix},
\quad
v_1=
\begin{bmatrix}
2\\1\\0.5
\end{bmatrix},
\quad
q_1=
\begin{bmatrix}
1\\1\\0.5
\end{bmatrix},
\quad
\alpha_1=0.5,
\quad
\beta_1=0.8
$$

$$
k_2=
\begin{bmatrix}
0\\1\\0
\end{bmatrix},
\quad
v_2=
\begin{bmatrix}
1\\3\\-1
\end{bmatrix},
\quad
q_2=
\begin{bmatrix}
1\\-1\\2
\end{bmatrix},
\quad
\alpha_2=0.25,
\quad
\beta_2=0.5
$$

Gated DeltaNet 单步公式：

$$
S_t=S_{t-1}\alpha_t(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top
$$

输出：

$$
o_t=S_tq_t
$$

## 1. 单步递推：token 1

先算 key 外积：

$$
k_1k_1^\top=
\begin{bmatrix}
1 & 0 & 0\\
0 & 0 & 0\\
0 & 0 & 0
\end{bmatrix}
$$

所以：

$$
I-\beta_1k_1k_1^\top=
\begin{bmatrix}
0.2 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}
$$

旧状态经过衰减和擦除：

$$
\alpha_1S_0(I-\beta_1k_1k_1^\top)
=0.5
\begin{bmatrix}
0.2 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
0.1 & 0 & 0\\
0 & 0.5 & 0\\
0 & 0 & 0.5
\end{bmatrix}
$$

新写入项：

$$
\beta_1v_1k_1^\top
=0.8
\begin{bmatrix}
2\\1\\0.5
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0
\end{bmatrix}
=
\begin{bmatrix}
1.6 & 0 & 0\\
0.8 & 0 & 0\\
0.4 & 0 & 0
\end{bmatrix}
$$

所以：

$$
S_1=
\begin{bmatrix}
1.7 & 0 & 0\\
0.8 & 0.5 & 0\\
0.4 & 0 & 0.5
\end{bmatrix}
$$

输出：

$$
o_1=S_1q_1=
\begin{bmatrix}
1.7 & 0 & 0\\
0.8 & 0.5 & 0\\
0.4 & 0 & 0.5
\end{bmatrix}
\begin{bmatrix}
1\\1\\0.5
\end{bmatrix}
=
\begin{bmatrix}
1.7\\1.3\\0.65
\end{bmatrix}
$$

## 2. 单步递推：token 2

第二个 key 外积：

$$
k_2k_2^\top=
\begin{bmatrix}
0 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 0
\end{bmatrix}
$$

所以：

$$
I-\beta_2k_2k_2^\top=
\begin{bmatrix}
1 & 0 & 0\\
0 & 0.5 & 0\\
0 & 0 & 1
\end{bmatrix}
$$

旧状态经过衰减和擦除：

$$
\alpha_2S_1(I-\beta_2k_2k_2^\top)
=
0.25
\begin{bmatrix}
1.7 & 0 & 0\\
0.8 & 0.5 & 0\\
0.4 & 0 & 0.5
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0\\
0 & 0.5 & 0\\
0 & 0 & 1
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
0.425 & 0 & 0\\
0.2 & 0.0625 & 0\\
0.1 & 0 & 0.125
\end{bmatrix}
$$

新写入项：

$$
\beta_2v_2k_2^\top
=0.5
\begin{bmatrix}
1\\3\\-1
\end{bmatrix}
\begin{bmatrix}
0 & 1 & 0
\end{bmatrix}
=
\begin{bmatrix}
0 & 0.5 & 0\\
0 & 1.5 & 0\\
0 & -0.5 & 0
\end{bmatrix}
$$

所以：

$$
S_2=
\begin{bmatrix}
0.425 & 0.5 & 0\\
0.2 & 1.5625 & 0\\
0.1 & -0.5 & 0.125
\end{bmatrix}
$$

输出：

$$
o_2=S_2q_2=
\begin{bmatrix}
0.425 & 0.5 & 0\\
0.2 & 1.5625 & 0\\
0.1 & -0.5 & 0.125
\end{bmatrix}
\begin{bmatrix}
1\\-1\\2
\end{bmatrix}
=
\begin{bmatrix}
-0.075\\-1.3625\\0.85
\end{bmatrix}
$$

单步 recurrent reference 的最终答案：

$$
O=
\begin{bmatrix}
1.7 & 1.3 & 0.65\\
-0.075 & -1.3625 & 0.85
\end{bmatrix}
$$

$$
S_{\text{next}}=
\begin{bmatrix}
0.425 & 0.5 & 0\\
0.2 & 1.5625 & 0\\
0.1 & -0.5 & 0.125
\end{bmatrix}
$$

## 12. KDA：Kimi Delta Attention 的计算推理优化

这一章对应 `tech_report.pdf` 里的 Kimi Delta Attention。KDA 可以理解成：

```text
KDA = Gated DeltaNet + channel-wise decay + 更省的 DPLR/chunkwise 实现
```

GDN 的遗忘门 $\alpha_t$ 是一个 scalar，通常每个 head 一个值。KDA 把它改成按 channel 的向量：

$$
\alpha_t\in\mathbb{R}^{d_k}
$$

也就是每个 key/channel 维度都有自己的遗忘速度。

### 12.1 先注意 state 方向

Kimi Linear 技术报告里的 state 方向和前面 GDN 手算例子是转置关系。

前面这份 GDN 例子为了方便写成：

$$
S\in\mathbb{R}^{d_v\times d_k},\quad o_t=S_tq_t
$$

Kimi 技术报告使用：

$$
S\in\mathbb{R}^{d_k\times d_v},\quad o_t=S_t^\top q_t
$$

两种写法本质一样，只是转置。下面讲 KDA 时跟随 Kimi 技术报告：

```text
q, k: [dk]
v:    [dv]
S:    [dk, dv]
o:    [dv]
```

### 12.2 KDA 的 recurrent 公式

KDA 的核心递推是：

$$
S_t=
(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_tk_tv_t^\top
$$

输出：

$$
o_t=S_t^\top q_t
$$

和 GDN 对比：

$$
\text{GDN:}\quad S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top
$$

$$
\text{KDA:}\quad S_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}+\beta_tk_tv_t^\top
$$

区别是：

- GDN：$\alpha_t$ 是 scalar，整个 state 一起 decay。
- KDA：$\operatorname{Diag}(\alpha_t)$ 是 $d_k\times d_k$ 对角矩阵，每个 key/channel 维度单独 decay。

直觉上，GDN 是“一整张记忆表一起褪色”；KDA 是“记忆表的每个 key 维度有自己的褪色速度”。这会带来更细粒度的位置感和 recency bias。

### 12.3 单步语义：先 decay，再 delta erase/write

把 KDA 单步拆开：

$$
\tilde{S}_{t-1}=\operatorname{Diag}(\alpha_t)S_{t-1}
$$

然后做 DeltaNet 风格的擦写：

$$
S_t=(I-\beta_tk_tk_t^\top)\tilde{S}_{t-1}+\beta_tk_tv_t^\top
$$

展开：

$$
S_t=\tilde{S}_{t-1}
-\beta_tk_t(k_t^\top\tilde{S}_{t-1})
+\beta_tk_tv_t^\top
$$

合并：

$$
S_t=\tilde{S}_{t-1}
+\beta_tk_t(v_t-\tilde{S}_{t-1}^\top k_t)^\top
$$

其中：

$$
\tilde{S}_{t-1}^\top k_t\in\mathbb{R}^{d_v}
$$

是 decayed state 里当前 key 对应的旧 value。KDA 单步可以理解成：

```text
1. 每个 key/channel 维度按 alpha_t 独立 decay
2. 用 k_t 从 decayed state 读旧 value
3. 写入 beta_t * (v_t - old_value)
4. 写入方向是 k_t
```

### 12.4 为什么 KDA 更像位置编码

Kimi 技术报告说 GDN 可以看成一种 data-dependent multiplicative positional encoding，而 KDA 更进一步。

如果忽略 delta 写入，只看 decay，GDN 的旧 state 到第 $r$ 个位置会乘：

$$
\gamma_r=\prod_{j=1}^r\alpha_j
$$

这是 scalar。

KDA 里每个 channel 都有自己的 decay：

$$
\gamma_r=
\prod_{j=1}^r\alpha_j
\in\mathbb{R}^{d_k}
$$

所以第 $c$ 个 channel 有：

$$
\gamma_{r,c}=\prod_{j=1}^r\alpha_{j,c}
$$

这类似 RoPE 的“不同维度有不同频率”，但 KDA 的频率/衰减是 data-dependent、learned，而且不要求正交旋转。

### 12.5 chunkwise KDA 的核心 shape

设一个 chunk：

```text
C  = chunk size
dk = key/query dim
dv = value dim
```

KDA 的 chunk 输入：

```text
Q:     [C, dk]
K:     [C, dk]
V:     [C, dv]
G:     [C, dk]    # log cumulative decay, 或者 cumulative gamma
beta:  [C]
S:     [dk, dv]
O:     [C, dv]
```

技术报告的 PyTorch-style pseudocode 里用 `g = g.cumsum(-2)`，所以实际 kernel 常常存的是 log decay：

$$
g_r=\sum_{j=1}^{r}\log \alpha_j
$$

于是区间 decay 用：

$$
\exp(g_r-g_i)
$$

这比直接存 $\gamma_r/\gamma_i$ 更稳。

### 12.6 KDA 的 W/U/D 关系

KDA 仍然保留 DeltaNet 的 $W,U,D$ 思路：

```text
W: erase key representation
U: write value representation
D: effective pseudo-value = U - W @ S
```

技术报告里的核心形式可以理解成：

$$
W=M(\Gamma^{1\to C}\odot K)
$$

$$
U=MV
$$

其中 $M$ 是 lower-triangular solve 得到的矩阵，类似 GDN/DeltaNet 里的 $T$：

$$
M=
\left[
I+\operatorname{StrictTril}\left(
\operatorname{Diag}(\beta)
\left(
\Gamma\odot KK^\top
\right)
\right)
\right]^{-1}
\operatorname{Diag}(\beta)
$$

不用被 inverse 吓到：实现不是求通用逆矩阵，而是 unit lower-triangular forward substitution。

KDA 的 effective pseudo-value：

$$
D=U-WS
$$

注意这里因为 Kimi 公式使用 $S:[d_k,d_v]$，所以是：

$$
W S:[C,d_k]\times[d_k,d_v]\rightarrow[C,d_v]
$$

前面 GDN 手算例子用的是 $S:[d_v,d_k]$，所以那里写成：

$$
D=U-\overleftarrow{W}S_0^\top
$$

这两个是同一件事的转置版本。

### 12.7 KDA 的输出计算

技术报告里的输出公式可以按两部分理解：

$$
O=O_{\text{inter}}+O_{\text{intra}}
$$

inter-chunk 部分：读 chunk 开始前的 state。

$$
O_{\text{inter}}=(\Gamma^{1\to C}\odot Q)S
$$

shape：

$$
[C,d_k]\times[d_k,d_v]\rightarrow[C,d_v]
$$

intra-chunk 部分：读当前 chunk 内已经写入的 pseudo-value。

$$
O_{\text{intra}}
=
\operatorname{Tril}
\left(
(\Gamma\odot Q)K^\top
\right)D
$$

shape：

$$
[C,C]\times[C,d_v]\rightarrow[C,d_v]
$$

整体就是：

```text
O = 读旧 state + 读 chunk 内新写入
```

这和 GDN 的：

$$
O=\overleftarrow{Q}S_0^\top+((QK^\top)\odot\Gamma)D
$$

是同一个 mental model，只是 KDA 的 $\Gamma$ 是 channel-wise 的，不再是简单 scalar mask。

### 12.8 KDA 的 state update

chunk 结束后，state 先整体按最后位置的 channel-wise decay 衰减：

$$
S\leftarrow \operatorname{Diag}(\gamma_C)S
$$

然后加上当前 chunk 写入：

$$
S\leftarrow S+(\Gamma^{i\to C}\odot K)^\top D
$$

shape：

$$
(\Gamma^{i\to C}\odot K)^\top:
[d_k,C]
$$

$$
D:[C,d_v]
$$

所以：

$$
[d_k,C]\times[C,d_v]\rightarrow[d_k,d_v]
$$

这就是 decode cache / recurrent state 的更新。推理时每层每 head 只需要保留固定大小：

$$
S\in\mathbb{R}^{d_k\times d_v}
$$

技术报告里常用 $d_k=d_v=128$，所以每个 head 的 recurrent state 是：

$$
128\times128
$$

### 12.9 KDA 相比通用 DPLR 为什么更快

通用 DPLR 可以写成类似：

$$
S_t=(D_t-a_tb_t^\top)S_{t-1}+k_tv_t^\top
$$

它表达力强，但 chunkwise 时会出现更多 decay ratio 和更多二级矩阵乘法，尤其 fine-grained decay 下容易有数值稳定问题，需要 secondary chunking。

KDA 的关键约束是把 DPLR 里的低秩方向绑定到 key 上，近似理解成：

```text
a_t = b_t = k_t
```

这样它仍然保留 delta rule 的 Householder-style 擦写结构，但减少了通用 DPLR 的额外计算。

技术报告总结的优化点：

- 避免部分 $1/\Gamma$ division 带来的数值不稳定。
- secondary chunking 从更多步骤减少到更少步骤。
- inter-chunk 和 output 计算里少了多次矩阵乘法。
- kernel 速度相比通用 DPLR 接近 2x。

### 12.10 推理优化：prefill 和 decode

KDA 推理分两种阶段。

prefill 阶段：

```text
输入一整段 prompt
按 chunk 做并行
每个 chunk 内用矩阵乘法
chunk 之间传递 state
```

适合使用上面的 chunk kernel：

```text
QK^T:      [C,dk] @ [dk,C]
A @ D:     [C,C] @ [C,dv]
K.T @ D:   [dk,C] @ [C,dv]
```

decode 阶段：

```text
每次只来一个新 token
直接用 recurrent update
不需要扫历史 KV
```

单步 decode 伪代码：

```python
# Kimi/KDA orientation
# S: [dk, dv]
# q, k, alpha: [dk]
# v: [dv]

S_decay = alpha[:, None] * S
old = S_decay.T @ k              # [dv]
d = beta * (v - old)             # [dv]
S = S_decay + k[:, None] @ d[None, :]
o = S.T @ q                      # [dv]
```

这就是 KDA 长上下文 decode 快的原因：标准 attention 每步要读越来越长的 KV cache，而 KDA 每步只读写固定大小 state。

### 12.11 和 GDN kernel 的实现差异

如果你已经写了 GDN kernel，改 KDA 时主要变这几处：

1. $\alpha$ 从 scalar 变成 vector：

```text
GDN alpha: [C]
KDA alpha: [C, dk]
```

2. $\gamma$ 从 scalar prefix product 变成 per-channel prefix product：

```text
GDN gamma: [C]
KDA gamma/log_gamma: [C, dk]
```

3. 所有 decay ratio 都可能是 vector：

```text
exp(g_r - g_i): [dk]
```

4. score 不是简单的：

$$
(QK^\top)\odot\Gamma
$$

而是先把 $Q$ 或 $K$ 按 channel-wise decay 缩放后再点积：

$$
A_{r,i}=q_r^\top\left(\exp(g_r-g_i)\odot k_i\right)
$$

5. state update 也不是 scalar-decayed key：

$$
\overrightarrow{k}_i=(\gamma_C/\gamma_i)k_i
$$

而是 channel-wise：

$$
\overrightarrow{k}_i=\exp(g_C-g_i)\odot k_i
$$

### 12.12 最小实现路线

建议按这个顺序写 reference：

1. 写 KDA single-step recurrent reference。
2. 写 chunkwise PyTorch reference，先用 fp32 和小 shape。
3. 只用 $C=2,d_k=d_v=3$ 之类的小例子对齐单步输出。
4. 再扩到 $C=64,d_k=d_v=128$。
5. 最后写 Triton/CUDA kernel，重点优化：

```text
KDA score:
  A[r, i] = dot(q[r] * exp(g[r] - g[i]), k[i])

pseudo-value:
  D = U - W @ S

output:
  O = (q * exp(g)) @ S + A @ D

state update:
  S = exp(g_C)[:, None] * S + (k * exp(g_C - g)).T @ D
```

如果写 CUDA kernel，第一版不必追求完全融合。先分成：

```text
1. compute log_gamma / decay ratios
2. compute M/T lower-triangular solve
3. compute W/U
4. compute D
5. compute O and S_next
```

等正确性对齐后，再考虑把 decay scaling 融合进 load，把 $A$、$D$、state update 放进 shared/register tile。

后面的 chunkwise 算法必须和这个结果完全一致。

## 2.5 从单步递推推到 chunkwise 公式

这一节先不代数字，只推公式。目标是解释后面这些东西为什么会出现：

$$
\gamma,\quad \Gamma,\quad D,\quad \overleftarrow{Q},\quad \overrightarrow{K}
$$

### 2.5.1 把单步公式改写成“旧状态 + 有效写入”

单步公式：

$$
S_t=S_{t-1}\alpha_t(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top
$$

因为 $\alpha_t$ 是标量，可以写成：

$$
S_t=\alpha_tS_{t-1}-\alpha_t\beta_tS_{t-1}k_tk_t^\top+\beta_tv_tk_t^\top
$$

合并后两项：

$$
S_t=\alpha_tS_{t-1}+\beta_t(v_t-\alpha_tS_{t-1}k_t)k_t^\top
$$

定义当前 token 的 effective write：

$$
d_t=\beta_t(v_t-\alpha_tS_{t-1}k_t)\in\mathbb{R}^{d_v}
$$

那么：

$$
S_t=\alpha_tS_{t-1}+d_tk_t^\top
$$

直觉：先把旧状态整体衰减，再把“新 value 减掉旧 state 在当前 key 下读出来的旧 value”写入当前 key 方向。

### 2.5.2 展开两个 token

第 1 个 token：

$$
S_1=\alpha_1S_0+d_1k_1^\top
$$

第 2 个 token：

$$
S_2=\alpha_2S_1+d_2k_2^\top
$$

把 $S_1$ 代进去：

$$
S_2=\alpha_2(\alpha_1S_0+d_1k_1^\top)+d_2k_2^\top
$$

所以：

$$
S_2=\alpha_1\alpha_2S_0+\alpha_2d_1k_1^\top+d_2k_2^\top
$$

定义：

$$
\gamma_1=\alpha_1,\quad \gamma_2=\alpha_1\alpha_2
$$

则：

$$
S_2=\gamma_2S_0+\frac{\gamma_2}{\gamma_1}d_1k_1^\top+d_2k_2^\top
$$

因为：

$$
\frac{\gamma_2}{\gamma_1}=\alpha_2
$$

这解释了为什么第 1 个 token 的写入传到 chunk 末尾时要乘 $\gamma_2/\gamma_1$。

### 2.5.3 输出为什么是 old + intra

第 $r$ 个输出是：

$$
o_r=S_rq_r
$$

任意位置 $r$ 的 state 可以写成：

$$
S_r=\gamma_rS_0+\sum_{i=1}^{r}\frac{\gamma_r}{\gamma_i}d_ik_i^\top
$$

所以：

$$
o_r
=\gamma_rS_0q_r+
\sum_{i=1}^{r}\frac{\gamma_r}{\gamma_i}d_i(k_i^\top q_r)
$$

转成矩阵行向量写法：

$$
O
=\overleftarrow{Q}S_0^\top+
\left((QK^\top)\odot\Gamma\right)D
$$

其中：

$$
\overleftarrow{Q}_r=\gamma_rq_r
$$

$$
\Gamma_{r,i}=
\begin{cases}
\gamma_r/\gamma_i,& r\ge i\\
0,& r<i
\end{cases}
$$

$$
D=
\begin{bmatrix}
d_1^\top\\
d_2^\top\\
\cdots
\end{bmatrix}
$$

也就是：

$$
O_{\text{old}}=\overleftarrow{Q}S_0^\top
$$

$$
O_{\text{intra}}=((QK^\top)\odot\Gamma)D
$$

### 2.5.4 $D$ 为什么不只是 $\operatorname{diag}(\beta)V$

注意 $d_t$ 里面有 $S_{t-1}$：

$$
d_t=\beta_t(v_t-\alpha_tS_{t-1}k_t)
$$

而 $S_{t-1}$ 已经包含前面 token 写入的内容。因此一般情况下，$d_t$ 会受到之前 $d_i$ 的影响。

把 $S_{t-1}$ 的展开代入：

$$
\alpha_tS_{t-1}k_t
=\gamma_tS_0k_t+
\sum_{i<t}\frac{\gamma_t}{\gamma_i}d_i(k_i^\top k_t)
$$

所以：

$$
d_t=\beta_tv_t
-\beta_t\gamma_tS_0k_t
-\beta_t\sum_{i<t}\frac{\gamma_t}{\gamma_i}d_i(k_i^\top k_t)
$$

这就是 chunkwise 算法需要 $T/W/U$ 的根本原因：每个 $d_t$ 不只依赖 $v_t$，还依赖之前所有 $d_i$ 和 key 相似度 $k_i^\top k_t$。

在这个例子里，$k_1^\top k_2=0$，两个 key 正交，所以交叉项刚好消失：

$$
d_1=\beta_1v_1-\beta_1\gamma_1S_0k_1
$$

$$
d_2=\beta_2v_2-\beta_2\gamma_2S_0k_2
$$

这就是后面能直接写：

$$
D=U-\overleftarrow{W}S_0^\top
$$

的原因。一般 key 不正交时，$T$ 会把那些交叉项也吸收进去。

### 2.5.4.1 $D$、$W$、$U$ 的关系

论文在 DeltaNet preliminary 里先定义了 $W$ 和 $U$。这两个量容易和这里的 $D$ 混在一起，可以这样区分：

- $W$ 负责表示“擦除方向”，也就是每个 token 怎样从旧 state 里减掉 $Sk$。
- $U$ 负责表示“写入内容”，也就是每个 token 怎样把 $v$ 写进 state。
- $D$ 是 effective write，表示“写入内容减去旧 state 擦除项”。

没有 gate 的 DeltaNet 单步公式：

$$
S_t=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top
$$

也就是：

$$
S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top
$$

如果 chunk 内所有 key 都互不影响，那么：

$$
W=\operatorname{diag}(\beta)K,\quad U=\operatorname{diag}(\beta)V
$$

这时：

$$
D=U-WS_0^\top
$$

因为第 $t$ 行：

$$
D_t=U_t-W_tS_0^\top=\beta_tv_t^\top-\beta_tk_t^\top S_0^\top
$$

转成列向量就是：

$$
d_t=\beta_t(v_t-S_0k_t)
$$

一般情况下，chunk 内前面的写入会影响后面的擦除，所以 $U,W$ 不能只是 $\operatorname{diag}(\beta)V$ 和 $\operatorname{diag}(\beta)K$，需要用 lower-triangular 的 $T$ 吸收依赖：

$$
T=\left[I+\operatorname{strictLower}(\operatorname{diag}(\beta)KK^\top)\right]^{-1}\operatorname{diag}(\beta)
$$

$$
W=TK,\quad U=TV
$$

shape 是：

$$
T\in\mathbb{R}^{C\times C},\quad
W\in\mathbb{R}^{C\times d_k},\quad
U\in\mathbb{R}^{C\times d_v}
$$

Gated DeltaNet 多了 $\alpha$ 衰减，所以要把 $W,Q,K,S$ 按位置乘上 $\gamma$ 或 $\gamma_C/\gamma_i$：

$$
\overleftarrow{W}_i=\gamma_i W_i
$$

$$
\overleftarrow{Q}_r=\gamma_r Q_r
$$

$$
\overrightarrow{K}_i=\frac{\gamma_C}{\gamma_i}K_i
$$

因此 Gated DeltaNet 里对应的是：

$$
D=U-\overleftarrow{W}S_0^\top
$$

$$
O=\overleftarrow{Q}S_0^\top+((QK^\top)\odot\Gamma)D
$$

$$
S_C=\gamma_CS_0+D^\top\overrightarrow{K}
$$

这里的核心关系就是：

$$
\boxed{D=U-\overleftarrow{W}S_0^\top}
$$

$U$ 是准备写入的新内容，$\overleftarrow{W}S_0^\top$ 是从旧 state 里读出来、需要被擦掉的旧内容，两者相减后才是当前 chunk 真正贡献给输出和下一状态的 effective write。

### 2.5.5 state update 为什么是 $D^\top\overrightarrow K$

chunk 末尾状态：

$$
S_C=\gamma_CS_0+\sum_{i=1}^{C}\frac{\gamma_C}{\gamma_i}d_ik_i^\top
$$

把所有 $d_i^\top$ 堆成：

$$
D\in\mathbb{R}^{C\times d_v}
$$

把所有衰减到 chunk 末尾的 key 堆成：

$$
\overrightarrow{K}_i=\frac{\gamma_C}{\gamma_i}k_i^\top
$$

则：

$$
D^\top\overrightarrow{K}
=
\sum_{i=1}^{C}d_i\left(\frac{\gamma_C}{\gamma_i}k_i^\top\right)
$$

所以：

$$
S_C=\gamma_CS_0+D^\top\overrightarrow K
$$

shape 对齐：

$$
D^\top\in\mathbb{R}^{d_v\times C},\quad
\overrightarrow K\in\mathbb{R}^{C\times d_k},\quad
D^\top\overrightarrow K\in\mathbb{R}^{d_v\times d_k}
$$

## 3. 把两个 token 组成 chunk

把两个 token 堆成矩阵：

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

shape：

$$
Q,K,V\in\mathbb{R}^{C\times d}=\mathbb{R}^{2\times 3}
$$

$$
S_0\in\mathbb{R}^{d_v\times d_k}=\mathbb{R}^{3\times 3}
$$

## 4. 衰减 gamma 到底是什么

chunk 内 cumulative decay：

$$
\gamma_1=\alpha_1=0.5
$$

$$
\gamma_2=\alpha_1\alpha_2=0.5\times0.25=0.125
$$

所以：

$$
\gamma=
\begin{bmatrix}
0.5\\
0.125
\end{bmatrix}
$$

decay-aware causal mask：

$$
\Gamma_{r,i}=
\begin{cases}
\gamma_r/\gamma_i,& r\ge i\\
0,& r<i
\end{cases}
$$

具体是：

$$
\Gamma=
\begin{bmatrix}
1 & 0\\
0.25 & 1
\end{bmatrix}
$$

## 5. 旧 state 的读取：left decay Q

初始 state 对第 $r$ 个输出的贡献要经过 $\gamma_r$：

$$
\overleftarrow{Q}=
\begin{bmatrix}
\gamma_1q_1^\top\\
\gamma_2q_2^\top
\end{bmatrix}
=
\begin{bmatrix}
0.5 & 0.5 & 0.25\\
0.125 & -0.125 & 0.25
\end{bmatrix}
$$

旧 state 直接读出来：

$$
O_{\text{old}}=\overleftarrow{Q}S_0^\top
=
\begin{bmatrix}
0.5 & 0.5 & 0.25\\
0.125 & -0.125 & 0.25
\end{bmatrix}
$$

shape：

$$
\overleftarrow{Q}S_0^\top:
[2,3]\times[3,3]\rightarrow[2,3]
$$

## 6. W、U、D 的数值计算

这个例子里 $k_1,k_2$ 正交，所以 $T$ 不引入额外 lower-triangular 修正，可以直接得到：

$$
W=
\begin{bmatrix}
\beta_1k_1^\top\\
\beta_2k_2^\top
\end{bmatrix}
=
\begin{bmatrix}
0.8 & 0 & 0\\
0 & 0.5 & 0
\end{bmatrix}
$$

$$
U=
\begin{bmatrix}
\beta_1v_1^\top\\
\beta_2v_2^\top
\end{bmatrix}
=
\begin{bmatrix}
1.6 & 0.8 & 0.4\\
0.5 & 1.5 & -0.5
\end{bmatrix}
$$

擦旧 state 的动作要带上当前位置对应的 decay：

$$
\overleftarrow{W}=
\begin{bmatrix}
\gamma_1 W_1\\
\gamma_2 W_2
\end{bmatrix}
=
\begin{bmatrix}
0.4 & 0 & 0\\
0 & 0.0625 & 0
\end{bmatrix}
$$

当前 chunk 的 effective write：

$$
D=U-\overleftarrow{W}S_0^\top
$$

因为 $S_0$ 是单位矩阵：

$$
D=
\begin{bmatrix}
1.6 & 0.8 & 0.4\\
0.5 & 1.5 & -0.5
\end{bmatrix}
-
\begin{bmatrix}
0.4 & 0 & 0\\
0 & 0.0625 & 0
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
1.2 & 0.8 & 0.4\\
0.5 & 1.4375 & -0.5
\end{bmatrix}
$$

shape：

$$
U:[2,3],\quad
\overleftarrow{W}S_0^\top:[2,3]\times[3,3]\rightarrow[2,3],\quad
D:[2,3]
$$

## 7. chunk 内 attention-like 读取

先算普通 score：

$$
QK^\top=
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}
$$

加上 causal 和 decay：

$$
A=(QK^\top)\odot\Gamma
=
\begin{bmatrix}
1 & 0\\
0.25 & -1
\end{bmatrix}
$$

chunk 内新写入贡献：

$$
O_{\text{intra}}=AD
$$

具体：

$$
O_{\text{intra}}=
\begin{bmatrix}
1 & 0\\
0.25 & -1
\end{bmatrix}
\begin{bmatrix}
1.2 & 0.8 & 0.4\\
0.5 & 1.4375 & -0.5
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
1.2 & 0.8 & 0.4\\
-0.2 & -1.2375 & 0.6
\end{bmatrix}
$$

shape：

$$
A D:[2,2]\times[2,3]\rightarrow[2,3]
$$

最终输出：

$$
O=O_{\text{old}}+O_{\text{intra}}
$$

$$
O=
\begin{bmatrix}
0.5 & 0.5 & 0.25\\
0.125 & -0.125 & 0.25
\end{bmatrix}
+
\begin{bmatrix}
1.2 & 0.8 & 0.4\\
-0.2 & -1.2375 & 0.6
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
1.7 & 1.3 & 0.65\\
-0.075 & -1.3625 & 0.85
\end{bmatrix}
$$

这和单步 recurrent 算出来的 $o_1,o_2$ 完全一致。

## 8. chunk 末尾 state 怎么更新

chunk 结束后，初始 state 本身要衰减到 chunk 末尾：

$$
\gamma_C S_0=0.125S_0=
\begin{bmatrix}
0.125 & 0 & 0\\
0 & 0.125 & 0\\
0 & 0 & 0.125
\end{bmatrix}
$$

第 $i$ 个 token 的写入传到 chunk 末尾，要乘：

$$
\gamma_C/\gamma_i
$$

所以 right-decayed key 是：

$$
\overrightarrow{K}=
\begin{bmatrix}
(\gamma_2/\gamma_1)k_1^\top\\
(\gamma_2/\gamma_2)k_2^\top
\end{bmatrix}
=
\begin{bmatrix}
0.25 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$$

state update：

$$
S_{\text{next}}=\gamma_CS_0+D^\top\overrightarrow{K}
$$

先算：

$$
D^\top=
\begin{bmatrix}
1.2 & 0.5\\
0.8 & 1.4375\\
0.4 & -0.5
\end{bmatrix}
$$

$$
D^\top\overrightarrow{K}
=
\begin{bmatrix}
1.2 & 0.5\\
0.8 & 1.4375\\
0.4 & -0.5
\end{bmatrix}
\begin{bmatrix}
0.25 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
0.3 & 0.5 & 0\\
0.2 & 1.4375 & 0\\
0.1 & -0.5 & 0
\end{bmatrix}
$$

因此：

$$
S_{\text{next}}=
\begin{bmatrix}
0.125 & 0 & 0\\
0 & 0.125 & 0\\
0 & 0 & 0.125
\end{bmatrix}
+
\begin{bmatrix}
0.3 & 0.5 & 0\\
0.2 & 1.4375 & 0\\
0.1 & -0.5 & 0
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
0.425 & 0.5 & 0\\
0.2 & 1.5625 & 0\\
0.1 & -0.5 & 0.125
\end{bmatrix}
$$

这也和单步递推的 $S_2$ 完全一致。

shape：

$$
D^\top\overrightarrow K:
[3,2]\times[2,3]\rightarrow[3,3]
$$

这里最能看出为什么用 $C=2,d=3$ 更清楚：state update 里的中间维度是 chunk size $C=2$，输出矩阵维度是 $d_v\times d_k=3\times3$。

## 9. 把这个例子翻译成 kernel 流程

对一个 chunk，kernel 里的数据 shape 是：

```text
Q:      [C, Dk] = [2, 3]
K:      [C, Dk] = [2, 3]
V:      [C, Dv] = [2, 3]
S:      [Dv,Dk] = [3, 3]
alpha:  [C]     = [2]
beta:   [C]     = [2]
O:      [C, Dv] = [2, 3]
S_next: [Dv,Dk] = [3, 3]
```

完整流程：

```text
1. gamma = cumulative_product(alpha)

2. Q_left[r] = gamma[r] * Q[r]

3. Gamma[r, i] = gamma[r] / gamma[i], only for r >= i

4. 计算 W 和 U
   简单正交例子里：
   W = diag(beta) @ K
   U = diag(beta) @ V

   一般情况里：
   T = inverse_lower(I + strict_lower(diag(beta) * (Gamma ⊙ K K^T))) * diag(beta)
   W = T @ K
   U = T @ V

5. W_left[r] = gamma[r] * W[r]

6. D = U - W_left @ S.T

7. A = causal((Q @ K.T) ⊙ Gamma)

8. O = Q_left @ S.T + A @ D

9. K_right[i] = gamma[C] / gamma[i] * K[i]

10. S_next = gamma[C] * S + D.T @ K_right
```

最重要的 shape 对齐：

```text
Q_left @ S.T:
  [2,3] @ [3,3] -> [2,3]

W_left @ S.T:
  [2,3] @ [3,3] -> [2,3]

A @ D:
  [2,2] @ [2,3] -> [2,3]

D.T @ K_right:
  [3,2] @ [2,3] -> [3,3]
```

## 10. 为什么论文里要用箭头

箭头只是为了说明“这个张量已经被衰减到哪个位置”。

可以这样记：

$$
\overleftarrow{Q}_r=\gamma_rQ_r
$$

这是从 chunk 开头的 state 衰减到第 $r$ 个 query。

$$
\overleftarrow{W}_r=\gamma_rW_r
$$

这是旧 state 被第 $r$ 个 token 擦除时需要带上的衰减。

$$
\overrightarrow{K}_i=\frac{\gamma_C}{\gamma_i}K_i
$$

这是第 $i$ 个 token 写入后继续传到 chunk 末尾的衰减。

$$
\Gamma_{r,i}=\frac{\gamma_r}{\gamma_i}
$$

这是第 $i$ 个 token 的写入影响第 $r$ 个 query 时经历的衰减。

所以，衰减并没有改变核心结构。它只是把 token 间的 $\alpha$ 累乘提前塞进 $Q,W,K$ 和 score mask 里，让 chunk 内还能写成矩阵乘法。

## 11. 写 kernel 时的最小实现建议

先写这个顺序，不要一开始追求融合：

```text
Kernel / function 1:
  计算 gamma 和 Gamma

Kernel / function 2:
  计算 KKT = K @ K.T
  计算 T
  计算 W = T @ K
  计算 U = T @ V

Kernel / function 3:
  计算 D = U - W_left @ S.T
  计算 O = Q_left @ S.T + A @ D
  计算 S_next = gamma_C * S + D.T @ K_right
```

等 reference 对齐后再融合：

- 不要 materialize `Q_left`，load `Q` 时乘 `gamma[r]`。
- 不要 materialize `K_right`，load `K` 时乘 `gamma_C / gamma[i]`。
- `Gamma` 如果 chunk 很小，可以放 shared memory。
- `T` 是 $C\times C$ lower triangular，小 chunk 时适合放 shared memory。
- `D.T @ K_right` 是 state update 的核心 GEMM-like 部分，后续优化重点在这里。

这个例子可以作为 kernel 单元测试。输入上面的数字，chunkwise kernel 应该输出：

$$
O=
\begin{bmatrix}
1.7 & 1.3 & 0.65\\
-0.075 & -1.3625 & 0.85
\end{bmatrix}
$$

$$
S_{\text{next}}=
\begin{bmatrix}
0.425 & 0.5 & 0\\
0.2 & 1.5625 & 0\\
0.1 & -0.5 & 0.125
\end{bmatrix}
$$
