# Kimi Linear: An Expressive, Efficient Attention Architecture

论文：Kimi Team，Technical Report，2025  
PDF：[Team et al. - 2025 - Kimi Linear An Expressive, Efficient Attention Architecture.pdf](/home/tpy/reps/papers/dnn/pdfs/Team%20et%20al.%20-%202025%20-%20Kimi%20Linear%20An%20Expressive,%20Efficient%20Attention%20Architecture.pdf)

## 论文一句话总结

Kimi Linear 的本质是：不要让每一层都背着完整 KV cache 跑长上下文，而是让大部分层用 Kimi Delta Attention（KDA）把历史压缩成固定大小的 recurrent state，只周期性保留少量 MLA full attention 层做精确全局检索。

KDA 是 Gated DeltaNet 的细粒度版本。它把每个 head 的历史写进一张 $[d_k,d_v]$ 的“记忆表” $S_t$，用 channel-wise gate 控制不同 feature 的遗忘速度，再用 delta rule 修正旧记忆。decode 时大部分层不扫历史 KV，只读写这张固定大小的表。

## 1. 背景和问题

标准 softmax attention 在长上下文推理里很贵：prefill 要处理 $S_q\times S_k$ 的注意力关系，decode 每生成一个 token 都要读完整历史 KV cache。上下文到 128k、512k、1M 时，瓶颈不只是算力，还有 KV cache 显存和 HBM 读取。Linear attention 可以把历史压成固定 state，但传统 linear attention 在语言建模、精确复制和长程检索上常弱于 full attention。Kimi Linear 要解决的是 attention 架构和推理效率问题：能否用 linear attention 替代大部分 full attention 层，同时保持甚至超过 full MLA 的质量。

## 2. 核心结论

- 在公平的 1.4T token 训练对比下，Kimi Linear 整体超过 full MLA baseline 和 hybrid GDN-H baseline。
- 长上下文 decode 中，Kimi Linear 最多减少约 75% token-level KV cache。
- 1M context 下，batch size=1 的 TPOT 相比 MLA 约 2.3x 更快；利用省下的显存扩大 batch 后，吞吐最高约 6.3x。
- 主要提升来自三个设计：KDA 的 channel-wise decay、delta rule 的纠错式 memory update、3:1 KDA/MLA inter-layer hybrid。
- 代价是实现复杂，需要专门 chunkwise KDA kernel；pure fixed-state linear attention 仍可能丢精确历史细节，所以论文保留 25% MLA 层。

## 3. 形象解释 / Mental Model

可以把标准 full attention 的 KV cache 想成一本不断变厚的笔记本。每生成一个 token，模型都要翻这本笔记本，把当前 query 和历史每一页的 key/value 对照一遍。上下文越长，笔记本越厚；decode 的每一步都要翻更多页。

Kimi Linear 的想法不是把翻笔记本这件事优化到极致，而是问：每一层真的都需要完整笔记本吗？KDA 的回答是：大部分层只保留一张固定大小的“索引卡” $S_t$。这张卡不是保存所有 token，而是把过去 token 的 key-value 关系压缩成一个矩阵 state。新 token 来了以后，KDA 做三件事：

1. 用 $\alpha_t$ 把旧索引卡上不同栏目按不同速度淡化；
2. 用当前 $k_t$ 找到索引卡里和当前 token 相关的方向；
3. 用 $v_t$ 修正这部分内容，然后用 $q_t$ 从索引卡里读出输出。

但只靠索引卡会丢细节。比如你要精确找“第 93721 个 token 后面那个变量名”，压缩 state 未必可靠。所以 Kimi Linear 每 4 层保留 1 层 MLA full attention：前三层用 KDA 压缩记忆，第四层仍然翻完整笔记本做全局检索。整体结构是 `KDA, KDA, KDA, MLA` 重复。

生成下一个 token 时，数据流可以这样想：

- 在 KDA 层：输入当前 hidden state，算出 $q_t,k_t,v_t,\alpha_t,\beta_t$，读取旧 state $S_{t-1}$，更新成 $S_t$，输出 $o_t$。这里 cache 是固定大小 $S_t:[B,H,d_k,d_v]$。
- 在 MLA 层：仍然把当前 token 的 K/V append 到 KV cache，然后 query 扫历史 K/V。这里 cache 仍随序列长度增长。
- 因为 4 层里只有 1 层 MLA，token-level KV cache 理论上减少 75%。

如果只记住一张图，应该记住 Figure 3：模型 block 里 token mixing 层按 3 个 KDA 加 1 个 MLA 交替，后面接 MoE。它说明 Kimi Linear 不是 pure linear attention，而是“固定 state 压缩 + 少量 full attention 精确检索”的混合架构。

## 4. 方法总览

标准 causal self-attention 参照：

```python
Q: [B, H, S, D]
K: [B, H, S, D]
V: [B, H, S, D]

scores = Q @ K.transpose(-1, -2)   # [B, H, S, S]
P = causal_softmax(scores)
O = P @ V                          # [B, H, S, Dv]
```

full attention 的特点：

- softmax score 来自 $QK^\top$；
- 每个 token 的历史信息以 token-level $K,V$ 形式保存；
- decode 第 $t$ 步需要读 $K_{\le t},V_{\le t}$；
- KV cache shape 近似 $[B,H,S,D]$，随 $S$ 线性增长。

KDA 的特点：

- 不计算 softmax，也不保存所有历史 K/V；
- 每个 head 保存 state $S_t\in\mathbb{R}^{d_k\times d_v}$；
- 输出是 $o_t=S_t^\top q_t$；
- state update 是带遗忘和纠错的 recurrent update；
- decode 复杂度不随历史长度线性增长，而主要取决于 $d_k,d_v$。

Kimi Linear 的模型架构：

- backbone 沿用 Moonlight 风格 MoE；
- 总参数 48B，activated parameters 3B；
- 每层 token mixing 后接 MoE channel mixing；
- KDA 和 MLA 按 3:1 交替；
- KDA 中 $d_k=d_v=128$；
- MLA 层使用 NoPE，不使用 RoPE；
- 输出侧使用 head-wise RMSNorm 和 sigmoid output gate。

这篇论文和标准 linear attention 的差别也很重要。很多 linear attention 写成 feature map 形式，比如 $\phi(q)^\top\phi(k)$，并维护分子/分母 prefix sum。KDA 不是这一路。KDA 更像 fast-weight memory：state $S_t$ 是一张可被在线更新的关联记忆表，delta rule 决定如何写入和纠错。

## 5. 关键公式 / 算法

### 5.1 Linear Attention 到 Delta Rule

**作用：**建立 KDA 的基础 mental model：attention 不一定要保存完整历史 KV，也可以把历史压进一个矩阵 state。

普通 linear attention 的 recurrent form：

$$
\begin{aligned}
S_t &= S_{t-1}+k_t v_t^\top \\
o_t &= S_t^\top q_t
\end{aligned}
$$

符号和 shape：

- $q_t,k_t\in\mathbb{R}^{d_k}$
- $v_t,o_t\in\mathbb{R}^{d_v}$
- $S_t\in\mathbb{R}^{d_k\times d_v}$

形象解释：$S_t$ 像一张 key 到 value 的记忆表。每来一个 token，就把 $k_t\rightarrow v_t$ 这条关联写进去。

问题是普通 linear attention 只会写，不会擦，也不会纠错。长序列里，旧关联、新关联、错误关联会互相干扰。

DeltaNet 把写入改成在线学习。它希望当前 state 在 key $k_t$ 上读出来的结果接近 $v_t$：

$$
L_t(S)=\frac{1}{2}\left\|S^\top k_t-v_t\right\|^2
$$

对 $S$ 做一步梯度下降：

$$
S_t
=S_{t-1}-\beta_t\nabla_S L_t(S_{t-1})
=\left(I-\beta_t k_tk_t^\top\right)S_{t-1}+\beta_t k_tv_t^\top
$$

直觉：先问旧记忆表“看到 $k_t$ 会读出什么”。如果读出的是 $\hat v_t=S_{t-1}^\top k_t$，而目标是 $v_t$，那就沿着 $k_t$ 方向把误差 $v_t-\hat v_t$ 写回去。

极小例子：假设 $d_k=2,d_v=3$，那么 $S$ 是 $2\times3$ 表。当前 $k_t$ 是一个二维方向，$k_t^\top S$ 会把这张表压成一个三维 value 预测。delta rule 只修正和 $k_t$ 方向相关的那部分表，而不是整张表无差别覆盖。

PyTorch 语义：

```python
def delta_step(q, k, v, beta, S):
    # q,k: [Dk], v: [Dv], S: [Dk, Dv]
    pred = k @ S                         # [Dv]
    residual = v - pred                  # [Dv]
    S = S + beta * torch.outer(k, residual)
    o = q @ S                            # [Dv]
    return o, S
```

影响：

- decode 不需要完整 KV cache，只需要 $S$；
- state 容量固定，长历史会被压缩；
- prefill 如果逐 token 扫会串行，所以后面需要 chunkwise 算法。

### 5.2 Kimi Delta Attention 的 Recurrent Formula

**作用：**KDA 在 Gated DeltaNet 上加入 channel-wise decay，让每个 feature channel 有独立遗忘速度。

GDN 的 forget gate 是 scalar：

$$
S_t=\alpha_t\left(I-\beta_t k_tk_t^\top\right)S_{t-1}+\beta_t k_tv_t^\top
$$

KDA 把 scalar $\alpha_t$ 换成 diagonal gate。结合正文 Eq. 1 和 §6.2 的 DPLR 重写，KDA 可理解为：

$$
\begin{aligned}
S_t
&=\left(I-\beta_t k_tk_t^\top\right)\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_t k_tv_t^\top \\
o_t&=S_t^\top q_t
\end{aligned}
$$

等价地：

$$
S_t
=\left(\operatorname{Diag}(\alpha_t)
-\beta_t k_tk_t^\top\operatorname{Diag}(\alpha_t)\right)S_{t-1}
+\beta_t k_tv_t^\top
$$

符号和 shape：

- $S_t\in\mathbb{R}^{d_k\times d_v}$
- $q_t,k_t\in\mathbb{R}^{d_k}$
- $v_t,o_t\in\mathbb{R}^{d_v}$
- $\alpha_t\in[0,1]^{d_k}$，每个 key channel 一个 decay
- $\operatorname{Diag}(\alpha_t)\in\mathbb{R}^{d_k\times d_k}$
- $\beta_t\in[0,1]$，控制 delta update 强度

形象解释：如果 $S_t$ 是一张记忆表，GDN 是“整张表统一变淡一点”，KDA 是“每一列/每个栏目按自己的速度变淡”。有些栏目保留长期信息，有些栏目快速忘记局部信息。

推导：DeltaNet 提供 rank-1 correction，GDN 增加 scalar decay，KDA 把 scalar decay 扩展成 diagonal matrix。§6.2 进一步说明它是 DPLR transition 的特殊情形。

极小例子：若 $d_k=4$，$\alpha_t=[0.99,0.95,0.5,0.1]$。第 1 个 channel 几乎保留长期记忆，第 4 个 channel 很快忘记。相比 head-wise scalar gate，KDA 能让同一个 head 同时维护长短不同的记忆。

PyTorch 语义：

```python
def kda_step(q, k, v, alpha, beta, S):
    # q,k,alpha: [Dk], v: [Dv], S: [Dk, Dv]
    S_decayed = alpha[:, None] * S
    pred = k @ S_decayed
    residual = v - pred
    S = S_decayed + beta * torch.outer(k, residual)
    o = q @ S
    return o, S
```

影响：

- softmax 被替换成 recurrent fast-weight update；
- 不使用传统 $\phi(q),\phi(k)$ feature map，也没有 softmax denominator；
- causal 性来自 state 只从过去递推；
- decode 从扫描 $[S,d]$ KV cache 变成读写 $[d_k,d_v]$ state；
- 精确检索能力受固定 state 容量限制，因此需要 MLA 层补足。

### 5.3 Chunkwise KDA：让 Prefill 仍然并行

**作用：**decode 可以一个 token 一个 token递推，但 prefill/训练不能这么慢。chunkwise KDA 把一段 token 打包处理，让 chunk 内用矩阵乘法并行，chunk 间才递推 state。

设 chunk size 为 $C$，第 $t$ 个 chunk：

- $Q_{[t]},K_{[t]}\in\mathbb{R}^{C\times d_k}$
- $V_{[t]}\in\mathbb{R}^{C\times d_v}$
- $\gamma_{[t]}^{1\to C}\in\mathbb{R}^{C\times d_k}$，每行是从 chunk 开头到当前位置的 cumulative channel-wise decay
- 初始 state $S^0_{[t]}\in\mathbb{R}^{d_k\times d_v}$

论文把 chunk 内第 $r$ 步后的 state 写成：

$$
S^r_{[t]}=P^r_{[t]}S^0_{[t]}+H^r_{[t]}
$$

这里 $P^r_{[t]}$ 是这个 chunk 前 $r$ 个 token 对旧 state 的总变换，$H^r_{[t]}$ 是 chunk 内新写入的内容。Appendix B 证明它们可以用 WY-like compact representation 表示，最终转成 $W_{[t]}$ 和 $U_{[t]}$ 两个 chunk 内辅助矩阵。

chunk 级 state update 的核心形式：

$$
S_{[t+1]}
=\operatorname{Diag}(\gamma^C_{[t]})S_{[t]}
+\left(\Gamma^{i\to C}_{[t]}\odot K_{[t]}\right)^\top
\left(U_{[t]}-W_{[t]}S_{[t]}\right)
$$

其中 $\Gamma^{i\to C}_{[t]}\in\mathbb{R}^{C\times d_k}$ 的第 $i$ 行是 $\gamma^C_{[t]}/\gamma^i_{[t]}$，也就是第 $i$ 个 token 的写入传到 chunk 末尾的 channel-wise 衰减。这里不能把 $\Gamma$ 当成普通 $C\times C$ scalar mask。

chunk 输出分成两部分：

$$
\begin{aligned}
O_{[t]}
&=\left(\Gamma^{1\to C}_{[t]}\odot Q_{[t]}\right)S_{[t]} \\
&\quad+\operatorname{Tril}\left(
\left(\Gamma^{1\to C}_{[t]}\odot Q_{[t]}\right)
\left(\frac{K_{[t]}}{\Gamma^{1\to C}_{[t]}}\right)^\top
\right)
\left(U_{[t]}-W_{[t]}S_{[t]}\right)
\end{aligned}
$$

这里 $\Gamma^{1\to C}_{[t]}$ 表示 $\gamma_{[t]}$ 的 $C\times d_k$ stack。输出里的 intra-chunk score 等价于：

$$
E_{r,i}
=q_r^\top\operatorname{Diag}(\exp(g_r-g_i))k_i,\quad r\ge i
$$

所以它是“按 channel 缩放后再点积”，不是先算 $QK^\top$ 再乘一个 scalar decay mask。

形象解释：decode 是逐条记账；prefill 是拿 64 条流水一起结算。chunk 内先算出这 64 条记录之间谁影响谁，再一次性更新跨 chunk 的总账 $S$。

小例子：假设 chunk size $C=4$。第 2 个 token 只能看第 1、2 个 token，第 4 个 token 可以看第 1 到 4 个 token。这就是一个 $4\times4$ lower-triangular interaction matrix。KDA 先在 chunk 内构造这个因果矩阵，再用矩阵乘法得到 4 个输出和下一个 chunk 的 state。

PyTorch 语义：

```python
def chunk_kda(q, k, v, log_decay, beta, init_state, C=64):
    # q,k: [B,T,H,Dk], v: [B,T,H,Dv]
    # after rearrange: [B,H,N,C,*]
    q, k, v, g, beta = rearrange_to_chunks(q, k, v, log_decay, beta, C)
    g = g.cumsum(dim=-2)

    B_kk = build_k_interactions(k, g)                         # [B,H,N,C,C]
    M = lower_triangular_solve_with_beta(B_kk, beta)           # [B,H,N,C,C]
    W = M @ (g.exp() * k)                                     # [B,H,N,C,Dk]
    U = M @ v                                                 # [B,H,N,C,Dv]

    S = init_state                                           # [B,H,Dk,Dv]
    outs = []
    for i in range(num_chunks):
        q_i, k_i, u_i, w_i, g_i = q[:,:,i], k[:,:,i], U[:,:,i], W[:,:,i], g[:,:,i]
        A_qk = build_causal_qk_interactions(q_i, k_i, g_i)    # [B,H,C,C]
        v_eff = u_i - w_i @ S                                # [B,H,C,Dv]
        o_i = (q_i * g_i.exp()) @ S + A_qk @ v_eff
        S = update_chunk_state(S, k_i, v_eff, g_i)
        outs.append(o_i)
    return concat_chunks(outs), S
```

影响：

- prefill/训练仍能并行；
- chunk 内大量计算是 GEMM，适合 Tensor Core；
- chunk 间保留 recurrent state 递推；
- 需要专门 CUDA/Triton kernel，普通 PyTorch 实现只能表达语义，不代表性能。

### 5.4 KDA 和 DPLR 的关系

**作用：**解释为什么 KDA 比通用 DPLR 更快。

通用 DPLR transition：

$$
S_t=(D-a_tb_t^\top)S_{t-1}+k_tv_t^\top
$$

论文 §6.2 的通用 DPLR 写法把写入项写成 $k_tv_t^\top$；和 KDA Eq. 1 对齐时，可以把这里的写入方向理解成已经吸收了 $\beta_t$。

KDA 是受约束的 DPLR：

$$
D=\operatorname{Diag}(\alpha_t),\qquad
a_t=\beta_t k_t,\qquad
b_t=k_t\odot\alpha_t
$$

所以：

$$
D-a_tb_t^\top
=\operatorname{Diag}(\alpha_t)
-\beta_t k_tk_t^\top\operatorname{Diag}(\alpha_t)
$$

形象解释：通用 DPLR 给你两支自由画笔 $a_t,b_t$，表达力强但每次画图都复杂；KDA 把两支画笔都绑定到 key $k_t$，自由度少一点，但能用更规整的计算路径快速跑。

论文声称 KDA 消除了 DPLR chunkwise 里的部分 secondary chunking 和多个矩阵乘法，Figure 2 中 KDA kernel 在 2K 到 64K 长度上接近 2x 快于 DPLR。

## 6. 数据流和实现设计

### 6.1 KDA 层内部数据流

每个 token 输入 $x_t\in\mathbb{R}^{d}$，每个 head 的 KDA 参数化：

$$
q_t^h,k_t^h
=\operatorname{L2Norm}(\operatorname{Swish}(\operatorname{ShortConv}(W_{q/k}^h x_t)))
\in\mathbb{R}^{d_k}
$$

$$
v_t^h
=\operatorname{Swish}(\operatorname{ShortConv}(W_v^h x_t))
\in\mathbb{R}^{d_v}
$$

$$
\alpha_t^h=f(W_\alpha^\uparrow W_\alpha^\downarrow x_t)\in[0,1]^{d_k},
\qquad
\beta_t^h=\operatorname{Sigmoid}(W_\beta^h x_t)\in[0,1]
$$

输出：

$$
o_t
=W_o\left(
\operatorname{Sigmoid}(W_g^\uparrow W_g^\downarrow x_t)
\odot
\operatorname{RMSNorm}(\operatorname{KDA}(q_t,k_t,v_t,\alpha_t,\beta_t))
\right)
$$

关键 shape：

```python
x:      [B, T, D_model]
q,k:    [B, T, H, Dk]
v:      [B, T, H, Dv]
alpha:  [B, T, H, Dk]
beta:   [B, T, H] or [B, T, H, 1]
state:  [B, H, Dk, Dv]
out:    [B, T, H, Dv]
```

论文实验里 $d_k=d_v=128$。

### 6.2 Prefill 怎么处理

prefill 是整段上下文一起进来。KDA 处理流程：

1. projection 得到 $Q,K,V,\alpha,\beta$；
2. 按 chunk size $C=64$ 切成 $N=T/C$ 个 chunks；
3. chunk 内构造 causal lower-triangular interactions；
4. 用 forward substitution 得到 compact $W,U$；
5. chunk 内并行算输出；
6. chunk 末尾更新 state，传给下一个 chunk。

计算复杂度，单 head：

$$
\operatorname{FLOPs}_{KDA}(T;C,d_h)
=6Td_h^2+3TCd_h+TC^2
$$

full attention 主导项：

$$
\operatorname{FLOPs}_{Attn}(T;d_h)=2T^2d_h
$$

因此长序列时，KDA prefill 近似线性增长，full attention 是二次增长。实际速度还取决于 kernel 是否把 chunk 内计算转成高效矩阵乘法。

### 6.3 Decode 怎么处理

decode 每步只来一个 token。

KDA 层：

```python
# per generated token, per KDA layer
q, k, v, alpha, beta = project(x_t)
o_t, S_t = kda_step(q, k, v, alpha, beta, S_prev)
cache = S_t
```

MLA 层：

```python
# per generated token, per MLA layer
append k_t, v_t to token-level KV cache
o_t = full_attention(q_t, K_cache, V_cache)
```

区别很直接：

- KDA cache 是 $[B,H,d_k,d_v]$，不随序列长度变；
- MLA cache 是 $[B,H,S,d]$，随 $S$ 增长；
- 3:1 hybrid 意味着只有 25% attention 层保留 token-level KV cache。

### 6.4 KV cache 和 PagedAttention / FlashInfer 的关系

Kimi Linear 不是 PagedAttention 或 FlashInfer 的替代品，而是处在不同层级。

- PagedAttention：仍然做 full attention，但优化 KV cache 的物理管理，减少碎片并支持 continuous batching。
- FlashInfer：仍然做 attention kernel，但统一处理 paged KV、block-sparse KV、prefix sharing、动态调度等。
- Kimi Linear：从模型架构上减少大部分层对完整 KV cache 的需求。

如果部署 Kimi Linear，KDA 层需要 fixed-state cache 管理；剩下 MLA 层仍然可以用 PagedAttention/FlashInfer 管理和计算 KV cache。

### 6.5 Softmax、normalizer、causal prefix

KDA 不使用 softmax，也没有 standard linear attention 中常见的分子/分母 normalizer。

传统 kernel linear attention 常像这样：

$$
o_t=\frac{\phi(q_t)^\top\sum_{i\le t}\phi(k_i)v_i^\top}
{\phi(q_t)^\top\sum_{i\le t}\phi(k_i)}
$$

KDA 则是：

$$
o_t=S_t^\top q_t
$$

其中 $S_t$ 通过 gated delta rule 递推更新。causal 性不是靠显式 softmax mask，而是因为 $S_t$ 只由 $1,\dots,t$ 的 token 更新；prefill 的 chunkwise 版本则用 lower-triangular matrix 保证 chunk 内 causal。

### 6.6 NoPE 和位置感

论文选择 MLA 层 NoPE。它的解释是：KDA/GDN 的 transition product 本身可以看作 data-dependent multiplicative positional encoding。

RoPE 的相对位置可以写成：

$$
s_{t,i}=q_t^\top\left(\prod_{j=i+1}^{t}R_j\right)k_i
$$

KDA/GDN 中，过去 token 对当前输出的影响也经过一串 transition：

$$
o_t
=\sum_{i=1}^{t}
\left(
q_t^\top
\prod_{j=i+1}^{t}
A_j\left(I-\beta_jk_jk_j^\top\right)
k_i
\right)v_i
$$

直觉上，RoPE 用固定旋转表达相对位置；KDA 用数据相关的遗忘和纠错表达“哪些历史该留下、留下多久”。channel-wise $\alpha_t$ 类似给不同 feature channel 配不同时间尺度。

## 7. 实验和效果

### 7.1 Synthetic tests

实验目的：验证 KDA 是否真的比 GDN/Mamba2 更会记忆、复制和检索。

任务：

- Palindrome：输入随机 token，输出反向序列，测试 exact copying。
- MQAR：multi-query associative recall，测试多位置 key-value 召回。
- Stack：模拟 64 个 LIFO stack，测试状态跟踪。

结果：KDA 在长度 256 到 2048 上准确率最好，在固定长度 1024 下收敛也更快。Mamba2 在这些设置下表现很差。含义是：只靠 multiplicative decay 不够，delta rule 的纠错式写入和 KDA 的细粒度遗忘都重要。

### 7.2 Ablation

表 1 比较 hybrid ratio、output gate、convolution。

关键结论：

- 3:1 KDA/MLA ratio 最好，validation PPL 5.65。
- 0:1 pure MLA validation PPL 5.77，在同训练预算下不如 Kimi Linear。
- 7:1 training PPL 接近，但 validation PPL 变差，说明 full attention 太少会损害泛化。
- 15:1 更差，说明 pure/near-pure linear 仍有 state 容量瓶颈。
- 去掉 output gate 或 convolution 都变差。
- Sigmoid output gate 明显优于 Swish output gate。

NoPE vs RoPE：Kimi Linear RoPE 短上下文接近，但长上下文平均更差。论文认为 RoPE 在少量 MLA 层里注入过强固定位置偏置，不利于长上下文扩展；NoPE 让 KDA 承担动态位置建模，更稳。

### 7.3 Scaling law

论文训练了 653M 到 1.7B activated params 的 MoE scaling-law 模型。KDA 模型保持 3:1 ratio，其余训练配置尽量和 MLA 一致。拟合结果显示 Kimi Linear 相比 MLA 有约 1.16x compute efficiency。这个数字不是夸张加速，但说明 KDA 不是某个单点规模的 trick。

### 7.4 1.4T 公平主实验

对比对象：

- full-attention MLA baseline；
- hybrid GDN-H baseline；
- Kimi Linear；
- 都是 48B total / 3B activated；
- 都用 1.4T tokens 训练配方。

Pretrain 短上下文：

- Kimi Linear 在 HellaSwag、ARC、MMLU、MMLU-Pro、TriviaQA、GSM8K、CRUXEval、CEval、CMMLU 等多数指标领先。
- MMLU-Pro：MLA 47.2，GDN-H 47.9，Kimi Linear 51.0。

SFT 后：

- Kimi Linear 在 BBH、MMLU、MMLU-Pro、MMLU-Redux、GPQA-Diamond、AIME、HMMT、PolyMath、LiveCodeBench 多数领先。
- 也有例外，例如 MATH500 和 EvalPlus 不是最高。

Long context 128k：

- RULER：MLA 81.3，GDN-H 80.5，Kimi Linear 84.3。
- MRCR：MLA 22.6，Kimi Linear 29.6。
- RepoQA：MLA 63.0，Kimi Linear 68.5。
- 平均分：MLA 52.2，GDN-H 51.2，Kimi Linear RoPE 51.8，Kimi Linear 54.5。

RL：

- 数学 RLVR 中，Kimi Linear 和 MLA 使用相同算法与超参。
- Kimi Linear 在训练集、MATH500、AIME 2025 曲线上都提升更快、更高。
- 这说明 Kimi Linear 不是只在静态 benchmark 上有效，在 long-form reasoning generation 下也有优势。

### 7.5 效率实验

Prefill：

- Kimi Linear 和 GDN-H 曲线几乎重合，说明 channel-wise decay 没带来明显额外 prefill latency。
- 从 128k 开始明显快于 MLA。
- 512k 约 2.3x，1M 约 2.9x。

Decode：

- batch size=1 下，1M context TPOT 约 2.3x 快于 MLA。
- 由于 KV cache 减少，可以支持更大 batch，图 1 报告 1M context 吞吐可达 6.3x。

Appendix D：

- 5.7T checkpoint 对比 Moonlight 5.7T，大多数 base/instruct benchmark 领先。
- Kimi Linear Instruct 在 RULER@128k 得 95.4，RULER@1M 得 94.8。
- 这说明 release model 的长上下文能力强，但 Appendix D 不是最严格架构公平对比，因为 Kimi Linear 是 48B total，而 Moonlight 是 16B total。

## 8. 我的判断

真正贡献：

- KDA：把 Gated DeltaNet 的 scalar decay 升级到 channel-wise diagonal decay，同时保持 chunkwise parallelism。
- 受约束 DPLR：没有追求最自由的 transition，而是选了一个足够表达、又能高效实现的子类。
- Hybrid 架构验证：3:1 KDA/MLA 在 48B MoE、1.4T tokens、SFT、long context、RL 上都做了比较完整的验证。
- 推理价值明确：它直接减少大部分层的 token-level KV cache，而不只是优化 KV cache 管理。

值得精读：

- §2.2：Linear attention、DeltaNet、GDN 的 fast-weight 解释。
- §3.1：chunkwise KDA。
- §4：KDA 参数化、output gate、3:1 hybrid。
- §6.1：KDA 作为 learnable positional mechanism。
- Appendix B/C：推导和伪代码。

可以跳过或粗读：

- Related Works 大部分；
- Appendix A 贡献列表；
- Appendix D 的大量 benchmark 表格只看结论即可。

价值判断：

- 对算法理解价值：高。
- 对推理系统价值：高。
- 对 kernel / 算子优化价值：中高。
- 是否值得复现：高，但应先复现数学语义，再看高性能 kernel。

## 9. 下一步阅读建议

建议下一步按这个顺序读原文：

1. §2.2：先理解 $S_t$ 是 fast-weight associative memory。
2. Eq. 1 和 §6.2：确认 KDA 是受约束 DPLR，不是普通 feature-map linear attention。
3. Figure 3：看清 KDA、MLA、MoE、gate 在模型中的位置。
4. Appendix C：按伪代码实现 chunkwise KDA。
5. Appendix B：对照伪代码理解 $P,H,W,U$ 的推导。
6. Table 1：理解为什么选 3:1、Sigmoid gate、NoPE。
7. Figure 7：理解 prefill/decode 速度收益分别来自哪里。

如果复现：

1. 先写 recurrent `kda_step`，验证单步 decode。
2. 写 naive recurrent 全序列 KDA，作为 correctness baseline。
3. 写 Appendix C 的 chunkwise PyTorch 版本，对比 naive recurrent 输出。
4. 固定 $C=64,d_k=d_v=128$，测试数值稳定性。
5. 跑 Palindrome 或 MQAR synthetic test。
6. 再看开源 `fla/ops/kda` kernel 和 vLLM 集成。

和前面几篇论文连起来看：

- PagedAttention：完整保留 KV cache，但把 KV cache 管理做成页表。
- FlashInfer：完整或稀疏访问 KV cache，但把 attention kernel 和调度做得更快。
- Kimi Linear：让大部分层不再需要完整 KV cache，只保留 fixed-size state。

这三条路线可以组合：Kimi Linear 的 KDA 层用 recurrent state，剩余 MLA 层仍然可以用 PagedAttention/FlashInfer 来优化 full attention。
