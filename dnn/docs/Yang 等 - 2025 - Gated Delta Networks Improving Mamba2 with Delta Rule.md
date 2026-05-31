# Yang 等 - 2025 - Gated Delta Networks: Improving Mamba2 with Delta Rule

## 论文一句话总结

Gated DeltaNet 把 Mamba2 的“按时间整体遗忘”能力和 DeltaNet 的“按 key 定点改写”能力合成一个线性 RNN 更新规则，让固定大小 recurrent state 同时具备快速清空无关记忆和精确更新 key-value 关联的能力。

核心公式是：

$$
S_t = S_{t-1}\alpha_t(I-\beta_t k_t k_t^\top)+\beta_t v_t k_t^\top,\quad
o_t=S_tq_t
$$

其中 $S_t\in\mathbb{R}^{d_v\times d_k}$ 是每个 head 的矩阵状态，$\alpha_t$ 是遗忘门，$\beta_t$ 是写入强度。

## 1. 背景和问题

标准 Transformer 的 softmax attention 在训练和推理里都要显式处理历史 KV，长上下文成本高。线性 attention / linear RNN 把历史压到固定大小状态 $S_t$，decode 可以从扫描完整 KV cache 变成读取 recurrent state，但代价是容量有限，长上下文和 retrieval 任务容易发生 memory collision。

Mamba2 的更新近似是：

$$
S_t=\alpha_t S_{t-1}+v_tk_t^\top
$$

它有 data-dependent decay，可以快速遗忘，但遗忘是“全局缩放”：如果只想忘掉某个 key-value 关联，所有记忆都会一起衰减。

DeltaNet 的更新是：

$$
S_t=S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_tv_tk_t^\top
$$

它能沿着当前 key 的方向先擦掉旧 value 再写入新 value，更适合 associative recall；但缺少强力 forget gate，遇到上下文切换或大量无关信息时，状态容易被塞满。

## 2. 核心结论

Gated DeltaNet 的结论很直接：Mamba2 的 gating 和 DeltaNet 的 delta rule 是互补的。

- $\alpha_t\to 0$ 时，模型可以快速清空旧状态，像 Mamba2 一样做 adaptive forgetting。
- $\alpha_t\to 1$ 时，模型退化成 DeltaNet，能对当前 key 方向做定点改写。
- 训练上，作者扩展 DeltaNet 的 WY / UT chunkwise 并行算法，把带 gate 的 delta rule 仍然转成大量矩阵乘法，适合 GPU / Tensor Core。
- 实验上，纯 Gated DeltaNet 在语言建模、常识推理、真实 retrieval、LongBench、长度外推上整体优于 Mamba2 和 DeltaNet；混合 SWA / Mamba2 后效果和吞吐继续提升。

代价是状态转移比 Mamba2 的标量/对角衰减更复杂，所以纯 Gated DeltaNet 吞吐略低于 Mamba2；但论文说相比 DeltaNet 几乎没有额外开销。

## 3. Mental Model

可以把线性 recurrent state $S_t$ 想成一张固定大小的“键值记忆表”，但它不是离散 KV cache，而是一个矩阵。每来一个 token，模型用 $k_t$ 作为地址方向，用 $v_t$ 作为内容，把内容写进矩阵里；查询时用 $q_t$ 去读这个矩阵。

普通线性 attention 像“只会追加笔记”：不断写 $v_tk_t^\top$，但旧内容永远不主动清理。

Mamba2 像“整页褪色”：每步先把整张记忆表乘上 $\alpha_t$，再写新内容。好处是能快速忘掉旧上下文，坏处是想擦一条记录时，其他记录也一起变淡。

DeltaNet 像“按索引覆盖”：当前 key $k_t$ 指到哪一列方向，就先用 $S_{t-1}k_t$ 读出旧 value，再用 $\beta_t(v_t-S_{t-1}k_t)k_t^\top$ 把这个方向改成新 value。好处是 retrieval 准，坏处是没有全局清场按钮。

Gated DeltaNet 就是“先判断要不要整页褪色，再按索引覆盖”。真正移动的瓶颈不是 softmax attention 的 $O(L^2)$ 计算，而是固定大小 recurrent state 的记忆管理能力：怎么在容量有限的矩阵里既保存重要 key-value，又过滤掉无关上下文。

如果只记一张图：记住一个 token 进入 block 后分成 $q,k,v,\alpha,\beta$ 五条路径，$q/k/v$ 经过 linear + short conv + SiLU，$q/k$ 做 L2 norm，然后用 gated delta rule 更新矩阵状态 $S$，最后输出再经过 norm、output gate 和 projection。

## 4. 方法总览

标准 attention 参照：

```python
Q: [B, H, S, D]
K: [B, H, S, D]
V: [B, H, S, D]

scores = Q @ K.transpose(-1, -2)
P = softmax(mask(scores))
O = P @ V
```

Gated DeltaNet 不保留完整历史 $K,V$，而是每个 batch/head 维护一个矩阵状态：

$$
S_t\in\mathbb{R}^{d_v\times d_k}
$$

单步 decode 时：

1. 当前 hidden state 投影得到 $q_t,k_t,v_t,\alpha_t,\beta_t$。
2. 用 $\alpha_t$ 对旧状态做整体 decay。
3. 用 $I-\beta_tk_tk_t^\top$ 在当前 key 方向擦除旧关联。
4. 用 $\beta_tv_tk_t^\top$ 写入新关联。
5. 输出 $o_t=S_tq_t$。

所以推理 cache 从标准 attention 的 $O(Ld)$ 历史 KV 变成 $O(d_vd_k)$ recurrent state。decode 理论上是常数状态，但状态矩阵读写成本和 head dim 相关。

训练时不能逐 token 串行跑 recurrence，否则 GPU 利用率差。论文把序列切成 chunk，每个 chunk 内通过 WY / UT 表示把 Householder-like transition 的连乘改写为矩阵乘法；chunk 间只传递最终状态 $S_{[t+1]}$。

## 5. 关键公式

### 5.1 Mamba2 的门控线性 attention

$$
S_t=\alpha_tS_{t-1}+v_tk_t^\top,\quad o_t=S_tq_t
$$

$\alpha_t\in(0,1)$ 是数据相关的 decay。展开后，每个历史 token 的贡献会被后续所有 $\alpha$ 的乘积衰减。直觉上，它能做快速遗忘，但不能区分“忘掉哪一条 key-value 关联”。

### 5.2 DeltaNet 的 delta rule

$$
S_t=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top
$$

也可以写成：

$$
S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top
$$

这里 $S_{t-1}k_t$ 是旧状态在当前 key 下读出来的 value，$v_t-S_{t-1}k_t$ 是 prediction error，$\beta_t$ 是学习率/写入强度。它等价于对在线回归损失做一步 SGD：

$$
L(S_t)=\frac{1}{2}\|S_tk_t-v_t\|^2
$$

### 5.3 Gated Delta Rule

$$
S_t=S_{t-1}\alpha_t(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top
$$

它可以理解为给 DeltaNet 的 test-time SGD 加了 adaptive weight decay。$\alpha_t$ 决定旧 fast weight 是否整体收缩，$\beta_t$ 决定当前 key 方向被改写多强。

简单 PyTorch 语义如下：

```python
# per batch/head, simplified
# S: [Dv, Dk], q/k: [Dk], v: [Dv]
old = S @ k                         # [Dv]
S = alpha * S
S = S - alpha * beta * old[:, None] @ k[None, :]
S = S + beta * v[:, None] @ k[None, :]
o = S @ q                           # [Dv]
```

实现时要注意论文公式里 $S_{t-1}\alpha_t(I-\beta_tkk^\top)$ 的顺序：先全局 decay，再沿当前 key 方向做 delta 擦写。

## 6. Chunkwise 训练和实现设计

DeltaNet 的难点是 chunk 内有一串：

$$
\prod_i(I-\beta_i k_i k_i^\top)
$$

直接算会串行。已有 DeltaNet 用 WY representation 和 UT transform 把它表示成低秩矩阵形式：

$$
P=I-W^\top K,\quad H=U^\top K
$$

并用三角矩阵 $T$ 递推出：

$$
W=TK,\quad U=TV
$$

Gated DeltaNet 的扩展是在 chunk 内加入 cumulative decay $\gamma$，也就是每个 token 相对 chunk 起点/终点有不同衰减系数。最终形式仍然保持类似：

$$
S_{[t+1]}=\overrightarrow{S}_{[t]}+
(\tilde{U}_{[t]}-W_{[t]}\overleftarrow{S}_{[t]}^\top K_{[t]})^\top
\overrightarrow{K}_{[t]}
$$

$$
O_{[t]}=\overleftarrow{Q}_{[t]}S_{[t]}^\top+
(Q_{[t]}K_{[t]}^\top\odot M)(\tilde{U}_{[t]}-\overleftarrow{W}_{[t]}S_{[t]}^\top)
$$

符号看起来复杂，但实现意义简单：把 token 级 recurrent update 变成 chunk 内矩阵乘法，chunk 之间传一个 state。这样训练仍然是线性复杂度，并且能吃 Tensor Core。

## 7. 架构细节

基础 Gated DeltaNet 沿用 Llama 风格宏结构：token mixer + SwiGLU MLP 堆叠，只是把 self-attention 换成 gated delta rule token mixer。

block 内部：

- $q,k,v$：linear projection -> short convolution -> SiLU。
- $q,k$：额外做 L2 normalization，论文消融显示这对稳定性和效果很重要。
- $\alpha,\beta$：只用 linear projection 生成。
- 输出：经过 normalization 和 output gate，再 output projection。

混合模型：

- Gated DeltaNet-H1：Gated DeltaNet + sliding window attention。
- Gated DeltaNet-H2：Mamba2 + Gated DeltaNet + sliding window attention。

SWA 补局部比较和局部 shift 能力，Gated DeltaNet 管长程压缩记忆，Mamba2 提供更高吞吐和补充的 gated recurrence。

## 8. 实验和效果

训练设置：主要对比 1.3B 参数模型，在 FineWeb-Edu 上训练 100B tokens，训练长度 4K；混合模型和 Samba 使用 2K sliding window。

### 语言建模和常识推理

在 Table 3 中，纯 recurrent 模型里 Gated DeltaNet 平均表现最好：

- Mamba2 平均 common-sense accuracy：54.89。
- DeltaNet：52.14。
- Gated DeltaNet：55.32。

混合模型继续提升：

- Gated DeltaNet-H1：56.40。
- Gated DeltaNet-H2：56.18。

### S-NIAH 检索分析

论文用 Single Needle-In-A-Haystack 解释为什么两种机制互补：

- S-NIAH-1 主要测长期保持，DeltaNet 很强，Mamba2 因 decay 在长序列下降明显。
- S-NIAH-2/3 加入真实 essay context 后，需要过滤无关信息；DeltaNet 因缺少清理机制下降，Mamba2 / Gated DeltaNet 更好。
- S-NIAH-3 的 value 是 UUID，更测复杂模式记忆；Gated DeltaNet 明显优于 Mamba2，说明 delta rule 对记忆复杂 value 有帮助。

### 真实 retrieval

Table 4 中，纯 recurrent 模型平均准确率：

- Mamba2：29.8。
- DeltaNet：26.2。
- Gated DeltaNet：30.6。

混合模型更强：

- Transformer++：37.0。
- Samba：37.3。
- Gated DeltaNet-H1：39.0。
- Gated DeltaNet-H2：40.1。

这说明纯线性 RNN 仍和 attention 有差距，但混合 attention 后可以超过纯 Transformer baseline。

### LongBench 和长度外推

LongBench 平均分：

- Mamba2：13.5。
- DeltaNet：13.6。
- Gated DeltaNet：16.6。
- Gated DeltaNet-H1：17.8。
- Gated DeltaNet-H2：18.4。

长度外推到 20K token 时，Gated DeltaNet 在 RNN 模型里整体 perplexity 最低；混合模型因为 SWA 分担局部上下文建模，表现更稳。

### 吞吐

Gated DeltaNet 和 DeltaNet 吞吐几乎相同；两者比 Mamba2 略慢，论文给出的差距大约是 2-3K tokens/sec，原因是 transition matrix 更有表达力也更复杂。混合 SWA 后，H1/H2 的训练吞吐反而优于纯 Gated DeltaNet，尤其 H1 在不同序列长度下都比较有竞争力。

## 9. 消融结论

400M / 15B tokens 消融显示：

- naive delta rule 明显更差，说明硬件友好的正规 chunkwise delta 不是小细节。
- short convolution 和 output gate 都重要。
- output norm 只有小幅收益。
- L2 norm 明显优于 L1 norm。
- feature map 里 SiLU 最好，但差距小于 normalization。
- head dim 128 是效果和效率的折中；head dim 256 perplexity 更低但计算更重。
- 混合层顺序里，Mamba2 + Gated DeltaNet + SWA 最好。

## 10. 我的判断

真正贡献有两个：

1. 算法层面：把 Mamba2 的 adaptive forgetting 和 DeltaNet 的 key-directed update 合到一个非常简洁的更新规则里。
2. 工程层面：证明这个规则不是只能串行跑，而是能扩展已有 WY / UT chunkwise 算法做高效训练。

理解价值：高。它把 linear attention、fast weight、test-time regression、Mamba2-style gating 放到了同一个 mental model 里。

推理系统价值：中到高。decode cache 是固定状态，长上下文潜力好；但真实 retrieval 仍需要混合 attention 才强。

kernel / 算子优化价值：高。核心在 chunkwise recurrent scan、三角矩阵、decay-aware masking、Tensor Core matmul 组织，非常适合研究 Flash Linear Attention 类 kernel。

是否值得复现：高。如果目标是理解现代 linear attention / Mamba 替代架构，建议先复现单层 PyTorch 语义，再看 FLA/Triton 实现。

## 11. 下一步阅读建议

如果已经看懂 Gated DeltaNet 的核心算法和 chunkwise 计算，这篇论文不需要从头精读，可以转成定点补读。

优先读：

- Section 3.1：gated delta rule 的公式和 online learning 解释。
- Section 3.3：chunkwise training 算法，这是工程核心。
- Figure 1：block design 和 H1/H2 混合结构。
- Table 2：最能解释为什么 gating 和 delta rule 互补。
- Appendix A：extended WY representation 的归纳证明。
- Appendix B.2：消融，尤其 L2 norm、short conv、output gate、head dim。

如果要实现，建议顺序是：

1. 写单步 recurrent PyTorch 版本，验证 shape 和 causal 输出。
2. 写 chunk 内 naive scan，对齐单步结果。
3. 再实现 WY/UT chunkwise 版本，对齐 naive scan。
4. 最后加入 $\alpha$ 的 chunk 内 cumulative decay。

如果目标是写 kernel，继续读论文的边际收益会下降。更值得做的是：

1. 写 naive recurrent PyTorch reference，先对齐单步公式。
2. 写 chunkwise PyTorch reference，对齐 recurrent 输出和 $S_{\text{next}}$。
3. 用极小 shape 例子做单元测试，例如 $d_k=d_v=2,C=2$。
4. 再写 CUDA/Triton forward kernel，先 fp32，再加 fp16/bf16 输入和 fp32 accumulation。
5. 最后再考虑 backward、融合 short conv / projection / gate。

读论文时可以跳过大段 related work 和 benchmark 描述；真正不能跳的是 block design 与消融，因为它们决定模型实现时哪些工程细节不能省。
