# KDA Kernel 简化版

这份笔记只保留实现 Kimi Delta Attention chunkwise kernel 需要的主线：

```text
单步 recurrent 语义 -> channel-wise decay -> chunkwise W/U/D -> 输出 -> state update -> kernel checklist
```

本文沿用论文和前面 Gated DeltaNet 手算例子的 state 方向：

$$
S_t\in\mathbb{R}^{d_k\times d_v},\quad
q_t,k_t,\alpha_t\in\mathbb{R}^{d_k},\quad
v_t,o_t\in\mathbb{R}^{d_v},\quad
o_t=S_t^\top q_t
$$

一个 chunk 内去掉 batch/head 维度后：

$$
Q,K,g,\gamma\in\mathbb{R}^{C\times d_k},\quad
V,U,D,O\in\mathbb{R}^{C\times d_v},\quad
\beta\in\mathbb{R}^{C},\quad
S_0,S_{\text{next}}\in\mathbb{R}^{d_k\times d_v}
$$

带回 batch/head 维度时通常是：

$$
q,k,g,\gamma:[B,H,T,d_k],\quad
v,o:[B,H,T,d_v],\quad
\beta:[B,H,T],\quad
state:[B,H,d_k,d_v]
$$

## 1. 单步语义

KDA 的 recurrent 公式对应论文 Eq. 1：

$$
S_t
=
\left(I-\beta_t k_tk_t^\top\right)\operatorname{Diag}(\alpha_t)S_{t-1}
+
\beta_tk_tv_t^\top
\in\mathbb{R}^{d_k\times d_v}
$$

$$
o_t=S_t^\top q_t\in\mathbb{R}^{d_v}
$$

拆成代码逻辑就是：

$$
\bar S_t=\operatorname{Diag}(\alpha_t)S_{t-1}
\in\mathbb{R}^{d_k\times d_v}
$$

$$
\hat v_t=\bar S_t^\top k_t
\in\mathbb{R}^{d_v}
$$

$$
d_t=\beta_t(v_t-\hat v_t)
\in\mathbb{R}^{d_v}
$$

$$
S_t=\bar S_t+k_td_t^\top
\in\mathbb{R}^{d_k\times d_v}
$$

最小 decode reference：

```python
# q, k, alpha: [dk]
# v, d, o:     [dv]
# S:           [dk, dv]
S_decay = alpha[:, None] * S
d = beta * (v - S_decay.T @ k)
S = S_decay + k[:, None] @ d[None, :]
o = S.T @ q
```

和 Gated DeltaNet 的区别只有一个核心点：GDN 的 $\alpha_t$ 是 scalar，KDA 的 $\alpha_t$ 是 $[d_k]$，所以 decay 发生在 state 的 key/channel 维度上。

## 2. Log Decay

论文 Appendix C 的伪代码输入 `g` 后会做：

```python
g = g.cumsum(-2)
```

所以实现里要区分两种 `g`：

$$
g^{raw}_r=\log\alpha_r\in\mathbb{R}^{d_k}
$$

$$
g_r=\sum_{j=1}^{r}g^{raw}_j
=\log\gamma_r
\in\mathbb{R}^{d_k}
$$

$$
\gamma_r=\exp(g_r)\in\mathbb{R}^{d_k}
$$

chunk 内第 $i$ 个 token 的写入传到第 $r$ 个 token，channel-wise 相对衰减是：

$$
\rho_{r,i}=\exp(g_r-g_i)\in\mathbb{R}^{d_k}
$$

这就是 KDA shape 变化的来源：GDN 里 $\rho_{r,i}$ 是 scalar，KDA 里 $\rho_{r,i}$ 是 $[d_k]$，不能再直接写成一个普通 $[C,C]$ mask 去乘 $QK^\top$ 或 $KK^\top$。

## 3. Chunkwise W/U/D

先构造 key-key interaction：

$$
B_{r,i}
=
k_r^\top\operatorname{Diag}(\rho_{r,i})k_i
=
\sum_{c=1}^{d_k}k_{r,c}\rho_{r,i,c}k_{i,c}
\in\mathbb{R}
$$

矩阵写法对应论文 Eq. 6：

$$
B=(\gamma\odot K)\left(\frac{K}{\gamma}\right)^\top
\in\mathbb{R}^{C\times C}
$$

然后做 lower-triangular solve：

$$
M=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
\right]^{-1}
\operatorname{Diag}(\beta)
\in\mathbb{R}^{C\times C}
$$

这里的 inverse 不是通用矩阵求逆，而是 unit lower-triangular forward substitution。

论文 Eq. 7 的 compact terms：

$$
W=M(\gamma\odot K)
\in\mathbb{R}^{C\times d_k}
$$

$$
U=MV
\in\mathbb{R}^{C\times d_v}
$$

effective pseudo-value：

$$
D=U-WS_0
\in\mathbb{R}^{C\times d_v}
$$

shape 对齐：

$$
WS_0:[C,d_k]\times[d_k,d_v]\to[C,d_v]
$$

直觉：

```text
W: 当前 chunk 对旧 state 的 erase/read 系数
U: 当前 chunk 内新 value 的 compact 写入
D: 扣掉旧 state 贡献后的有效写入 value
```

## 4. 输出

输出分成 inter-chunk 和 intra-chunk 两部分，对应论文 Eq. 9。

读 chunk 开始前的旧 state：

$$
O_{\text{inter}}=(\gamma\odot Q)S_0
\in\mathbb{R}^{C\times d_v}
$$

shape：

$$
(\gamma\odot Q)S_0:[C,d_k]\times[d_k,d_v]\to[C,d_v]
$$

读当前 chunk 内已经写入的 pseudo-value：

$$
E_{r,i}
=
\begin{cases}
q_r^\top\operatorname{Diag}(\rho_{r,i})k_i,&r\ge i\\
0,&r<i
\end{cases}
$$

矩阵写法：

$$
E=
\operatorname{Tril}\left(
(\gamma\odot Q)
\left(\frac{K}{\gamma}\right)^\top
\right)
\in\mathbb{R}^{C\times C}
$$

$$
O_{\text{intra}}=ED
\in\mathbb{R}^{C\times d_v}
$$

最终：

$$
O=O_{\text{inter}}+O_{\text{intra}}
=
(\gamma\odot Q)S_0+ED
\in\mathbb{R}^{C\times d_v}
$$

## 5. State Update

chunk 结束后，旧 state 先按 chunk 最后位置的 cumulative channel decay 衰减：

$$
\operatorname{Diag}(\gamma_C)S_0
\in\mathbb{R}^{d_k\times d_v}
$$

当前 chunk 的写入传到 chunk 末尾：

$$
K_{\text{right},i}
=
\rho_{C,i}\odot k_i
=
\exp(g_C-g_i)\odot k_i
\in\mathbb{R}^{d_k}
$$

$$
K_{\text{right}}\in\mathbb{R}^{C\times d_k}
$$

所以：

$$
S_{\text{next}}
=
\operatorname{Diag}(\gamma_C)S_0
+
K_{\text{right}}^\top D
\in\mathbb{R}^{d_k\times d_v}
$$

shape：

$$
K_{\text{right}}^\top D:[d_k,C]\times[C,d_v]\to[d_k,d_v]
$$

decode cache 每层每 head 只需要保留这个 fixed-size state。论文常用 $d_k=d_v=128$，所以单 head state 是：

$$
128\times128
$$

## 6. Kernel Checklist

最小实现流程：

```text
1. 读 Q,K,V,g_raw,beta,S0
   Q,K,g_raw: [C,dk], V: [C,dv], beta: [C], S0: [dk,dv]

2. 计算 g = cumsum(g_raw, dim=0), gamma = exp(g)
   g,gamma: [C,dk]

3. 计算 B[r,i] = dot(k[r] * exp(g[r]-g[i]), k[i])
   B: [C,C]

4. 计算 M = inv_lower(I + strict_lower(diag(beta) @ B)) @ diag(beta)
   M: [C,C]

5. 计算 W = M @ (gamma * K), U = M @ V
   W: [C,dk], U: [C,dv]

6. 计算 D = U - W @ S0
   D: [C,dv]

7. 计算 E[r,i] = dot(q[r] * exp(g[r]-g[i]), k[i]), r >= i
   E: [C,C]

8. 计算 O = (gamma * Q) @ S0 + E @ D
   O: [C,dv]

9. 计算 K_right = exp(g_C - g) * K
   K_right: [C,dk]

10. 计算 S_next = gamma_C[:,None] * S0 + K_right.T @ D
    S_next: [dk,dv]
```

和论文一致性核对：

```text
Eq. 1:  单步 recurrent 公式
Eq. 6:  B 和 M 的 lower-triangular solve
Eq. 7:  W = M(gamma*K), U = M V
Eq. 8:  S_next 更新
Eq. 9:  O = inter + intra 输出
Appendix C Listing 1: g.cumsum(-2) 和 exp(g_r-g_i) 的实现形式
```

写 reference 时，先用 fp32 小 shape 验证：

```text
C = 2 或 4
dk = dv = 3
随机 Q,K,V,alpha,beta,S0
chunkwise O/S_next 必须逐 token recurrent 对齐
```
