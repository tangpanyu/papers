# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

论文：Zihao Ye 等，MLSys 2025，arXiv:2501.01005v2

## 论文一句话总结

FlashInfer 的本质是给 LLM serving 做一个 attention engine：用 block-sparse / composable formats 统一各种 KV cache 访问关系，用 CUDA/CUTLASS + JIT 生成可定制 attention kernel，再用运行时 load-balanced scheduler 处理 batch 内动态序列长度。

它不是替代 PagedAttention 的内存管理，而是在 paged KV、radix prefix cache、tree attention、KV pruning 等复杂布局上高效计算 attention。

## 1. 背景和问题

LLM serving 里的 attention 比训练阶段更不规整。训练通常是大而规则的 dense attention；serving 同时存在 prefill、decode、incremental prefill，不同 request 的 query length 和 KV length 还在每步变化。KV cache 可能来自 PagedAttention page table、RadixAttention radix tree、shared prefix cache、speculative decoding tree、sliding window 或 pruning mask。

与此同时，模型侧 attention 变体也越来越多：GQA/MQA、RoPE fusion、ALiBi、logits soft cap、custom mask、非 softmax attention。每个 serving 框架都手写专用 kernel 会造成维护成本和性能覆盖问题；CUDA Graph 又要求 kernel launch 配置和指针稳定，和动态 serving workload 天然冲突。

FlashInfer 解决的是 kernel/runtime 层问题：如何把复杂 KV-cache layout、attention variants 和动态 batch 调度统一到一个高性能 attention 后端里。

## 2. 核心结论

论文证明：LLM serving 的 attention 可以抽象成“ragged query/output + block-sparse KV-cache + 可组合 attention state + 动态 scheduler”。在这个抽象下，PagedAttention、RadixAttention、prefix sharing、tree attention 和 fine-grained KV sparsity 都能被统一表达。

主要提升来自：

- 用 BSR/block-sparse 表示统一 KV-cache 访问关系。
- 用 composable formats 把 shared prefix 和 unique suffix 拆成不同 block size 的子问题，提升 KV 复用。
- 用 CUDA/CUTLASS 模板和 JIT functor 支持不同 attention 变体。
- 用 split-K 风格的 load-balanced scheduling 减少长短序列混 batch 时的 SM idle。
- 用 plan/run 和预分配 workspace 兼容 CUDA Graph。

代价是系统复杂度高：需要编译期模板、JIT cache、运行时 plan、workspace 管理和 contraction kernel。对于很规则的 dense prefill，FlashInfer 相比专用 dense kernel 不一定总是更简单；对于 vLLM bf16 集成，论文也提到 host/Python overhead 会抵消部分收益。

## 3. 形象解释 / Mental Model

可以把 LLM serving 的 attention 想成一个仓库拣货问题。每个 query 是一张订单，KV cache 是货架。PagedAttention 解决“货物怎么按页放、怎么避免仓库空洞”；FlashInfer 解决“订单要访问哪些货架、怎么组织拣货路线、怎么让工人负载均衡、特殊订单规则怎么快速换模板”。

baseline 通常是每种 KV layout 写一套 kernel：

```text
paged KV kernel
radix prefix kernel
tree attention kernel
sliding window kernel
custom mask kernel
...
```

FlashInfer 把它们都看成 block-sparse 矩阵：

```text
row block: 一组 query tiles
column block: 一组 KV-cache pages / blocks
nonzero block: 这组 query 需要访问这组 KV
```

一个 decode step 的小例子：

1. batch 里有多个 requests，每个 request 当前 query length 可能是 1、16 或更多。
2. 每个 request 的历史 KV length 不同，KV 物理位置也可能是 paged/radix/sparse。
3. serving 框架把这些访问关系转成 BSR indices：`indptr`、`indices`、page ids。
4. FlashInfer `plan()` 根据当前 lengths 和硬件 CTA 数量拆分长 KV，生成调度元数据。
5. `run()` 执行固定形态 CUDA kernel，产生 attention state。
6. 如果某个 query 的 KV 被拆成多个 chunks，再用 contraction kernel 通过 $\oplus$ 合并 partial states。

如果只记住一张图，应该记住 Figure 1：attention spec、task info、KV-cache layout 在编译/JIT 侧决定 kernel；sequence length 在运行时进入 scheduler；最终 kernel 读 KV cache、workspace，写 output。它说明 FlashInfer 真正移动的瓶颈是“复杂 serving attention 缺少统一高性能执行抽象”。

## 4. 方法总览

标准 attention 参照：

```python
Q: [B, H_q, S_q, D]
K: [B, H_kv, S_kv, D]
V: [B, H_kv, S_kv, D]

score = Q @ K.transpose(-1, -2) / sqrt(D)
score = mask_or_bias(score)
P = softmax(score, dim=-1)
O = P @ V
```

LLM serving 中常见 shape：

```text
decode:
  query length S_q 很小，常为 1
  KV length S_kv 很大，并且 batch 内不同 request 不同

prefill:
  S_q 接近 S_kv，计算更像 dense attention

GQA/MQA:
  H_q > H_kv
  group size g = H_q / H_kv
```

FlashInfer 改动的是执行抽象：

- Query/output 用 ragged tensor 存，不 padding。
- KV-cache 访问关系用 BSR/block-sparse 表示。
- Kernel 先把 sparse/scattered KV gather 到 shared memory，再用 dense tensor core 计算。
- Attention variants 用 transform/mask functor 注入模板。
- Runtime scheduler 按当前 sequence lengths 生成 split-K 调度。

和 PagedAttention 的关系：

```text
PagedAttention:
  负责 KV cache 怎么分配、共享、映射 logical block -> physical block

FlashInfer:
  负责在 page table/radix tree/sparse mask 等访问关系上怎么高效算 attention
```

复杂度层面，FlashInfer 不把 exact attention 的 $O(S_q S_{kv} D)$ 改成低阶；它通过更好的数据布局、复用、调度和融合提高硬件效率。

## 5. 关键公式 / 算法

### 5.1 Attention State Composition

作用：把同一个 query 对不同 KV chunks 的 partial attention 合并成全局 exact attention。这是 split-K、shared-prefix 分解和 partial output contraction 的数学基础。

对 query $q$ 和一组 KV index $I$，定义：

$$
\operatorname{LSE}(I) =
\log \sum_{i \in I} \exp(q^\top k_i)
$$

$$
O(I) =
\sum_{i \in I}
\frac{\exp(q^\top k_i)}
{\exp(\operatorname{LSE}(I))}
v_i
$$

attention state 是：

$$
S(I) =
\begin{bmatrix}
O(I) \\
\operatorname{LSE}(I)
\end{bmatrix}
$$

两个不相交 KV chunks $I, J$ 的合并为：

$$
S(I \cup J) = S(I) \oplus S(J)
$$

其中：

$$
O(I \cup J) =
\frac{
\exp(\operatorname{LSE}(I)) O(I) +
\exp(\operatorname{LSE}(J)) O(J)
}{
\exp(\operatorname{LSE}(I)) +
\exp(\operatorname{LSE}(J))
}
$$

$$
\operatorname{LSE}(I \cup J) =
\log(
\exp(\operatorname{LSE}(I)) +
\exp(\operatorname{LSE}(J))
)
$$

符号和 shape：单 query 单 head 下 $q,k_i,v_i \in \mathbb{R}^{D}$，$O(I) \in \mathbb{R}^{D}$，$\operatorname{LSE}(I)$ 是标量。batch/GQA 下这个 state 会扩展到 query tile、head 和 head_dim 维度。

直觉：每个 KV chunk 先给出“局部加权平均”和“局部总权重的 log”，合并时按两个 chunk 的真实 softmax mass 重新加权。因为 $\oplus$ 满足结合律和交换律，chunks 可以用任意顺序合并。

PyTorch 语义：

```python
def merge_state(o1, lse1, o2, lse2):
    m = torch.maximum(lse1, lse2)
    w1 = torch.exp(lse1 - m)
    w2 = torch.exp(lse2 - m)
    lse = m + torch.log(w1 + w2)
    out = (w1[..., None] * o1 + w2[..., None] * o2) / (w1 + w2)[..., None]
    return out, lse
```

影响：长 KV 可以拆成多个 chunks 分给不同 CTAs，最后 contraction 合并；shared prefix 和 unique suffix 也可以分开算再合并。

### 5.2 Block-Sparse KV-Cache Format

作用：把 page table、radix tree、sparse mask 等 KV-cache 访问关系统一成 BSR。

BSR 的概念形状：

```text
row block size Br: query tile size
column block size Bc: KV page/block size
indptr: [num_row_blocks + 1]
indices: [num_nonzero_blocks]
data/page ids: physical KV block ids
```

如果某个 row block 的 queries 需要访问第 2、7、8 个 KV blocks：

```text
indptr[row]   = p
indptr[row+1] = p + 3
indices[p:p+3] = [2, 7, 8]
```

真实 kernel 对每个 nonzero block 做：

```text
physical KV block id
  -> gather K/V from global memory
  -> pack into shared memory
  -> dense attention tile compute
  -> update attention state
```

直觉：BSR 是 FlashInfer 的“访问路线图”。它不要求 KV cache 物理连续，只要求告诉 kernel 哪些 query tiles 要读哪些 KV blocks。

### 5.3 Load-Balanced Scheduling

作用：解决 batch 内 sequence length 不均导致的 SM idle。长 request 的 KV work 多，短 request 的 KV work 少；如果一个 request 固定给一个 CTA，长请求会拖慢整轮 kernel。

调度语义：

```python
def plan(lengths, num_ctas):
    total_kv = sum(lengths)
    target = ceil(total_kv / num_ctas)
    work_items = []
    for req_id, kv_len in enumerate(lengths):
        for begin in range(0, kv_len, target):
            end = min(begin + target, kv_len)
            work_items.append((req_id, begin, end))
    return greedy_assign_to_ctas(work_items, num_ctas)
```

真实实现还要考虑 query tile size、head、workspace 上界和 CUDA Graph 兼容。被拆分的 request 会产生 partial attention states，之后用 $\oplus$ contraction 合并。

影响：uniform/skewed sequence length 下，调度器减少 SM 空转；同时通过固定 grid size、预分配 workspace 和 plan/run 拆分，尽量满足 CUDA Graph 的静态要求。

## 6. 数据流和实现设计

### 6.1 输入输出和 Layout

FlashInfer 的核心输入可以分为四类：

```text
attention specification:
  logits transform, mask, query/key/value transform, output transform

task information:
  heads, head_dim, dtype, GQA group size, tile size

KV-cache layout specification:
  dense/ragged/paged/block-sparse indices

runtime sequence length information:
  每个 request 的 query length 和 KV length
```

Query/output 是 ragged tensor，避免 padding：

```text
Q_data: [sum_i S_q_i, H_q, D]
qo_indptr: [batch + 1]
O_data: [sum_i S_q_i, H_q, D]
```

KV-cache 可以是 paged/block-sparse：

```text
K/V pages: [num_pages, H_kv, page_size, D]
BSR indices/indptr: 描述每个 query tile 访问哪些 pages
```

### 6.2 Prefill

prefill 的 $S_q$ 和 $S_{kv}$ 都较长，计算更接近训练阶段 dense attention。FlashInfer 可以用 ragged dense KV prefill API，也可以用 paged KV prefill API。

如果 KV 连续，Hopper 上可以用 TMA 等 FA3 路径；如果 KV 是 sparse/paged gather，TMA 受限于非仿射访问，论文附录报告 prefill sparse gather 相比 dense 大约有 10% gap。decode 中 sparse 和 dense gap 很小，因为本来主要受 KV 读取带宽限制。

### 6.3 Decode

decode 的 $S_q$ 很短，经常为 1。此时 GQA/MQA 的 KV 复用很重要。FlashInfer 使用 head-group fusion：把 query head 维度和 query length 维度融合，让同一 KV head 下多个 query heads 共享一次 K/V load。

数据流：

```text
Q tile
  -> 根据 BSR/page table 找到 KV pages
  -> sparse gather K/V 到 shared memory
  -> tensor core / CUDA core 计算 score
  -> logits transform / mask
  -> online softmax 更新 attention state
  -> 累加 V 得到 partial output
  -> 如有 split-K，用 contraction 合并 partial states
```

### 6.4 Composable Formats

单一 block-sparse format 需要固定 $B_r$。$B_r$ 大时，多个 queries 可以共享 K/V 的 shared memory/register load，但碎片更大；$B_r$ 小时更灵活，但复用差。

FlashInfer 用 composable formats 把一次 attention 拆成多个 sparse 子格式。例如 parallel generation 中，多条候选共享长 prefix，但 suffix 各不相同：

```text
shared prefix:
  用较大 Br，让同组 queries 共享 KV load

unique suffix:
  用较小 Br，减少碎片和无效访问
```

它不移动 KV cache，只重新构造 sparse indices 和 indptr。最后各子问题的 attention states 用 $\oplus$ 合并。

### 6.5 JIT Customization

FlashInfer 的 attention 模板支持用户注入：

- `QueryTransform`
- `KeyTransform`
- `ValueTransform`
- `OutputTransform`
- `LogitsTransform`
- `LogitsMask`

这让 RoPE fusion、logits soft cap、ALiBi、sliding window、custom mask、FlashSigmoid 等变体可以变成 fused kernel，而不是在 attention 前后额外 launch kernel。

论文选择 CUDA/CUTLASS 而不是只用 Triton，原因是需要更细的寄存器控制、PTX intrinsic、warp specialization、TMA 等 Hopper/Ampere 细节。

### 6.6 CUDA Graph 兼容

CUDA Graph 要求 captured kernel 的参数和指针稳定。FlashInfer 使用 inspector-executor 风格：

```text
plan():
  CPU 侧根据当前 sequence lengths 生成调度元数据
  写入预分配 pinned host/device workspace

run():
  使用固定 grid/workspace 指针执行 attention kernel
  可被 CUDA Graph capture/replay
```

workspace 需要预估上界，包括 scheduler metadata 和 split-K partial outputs。附录给出的 partial output 上界大致与 $2 \times \#CTA \times T_q \times H_q \times (D+1)$ 成正比，其中 $D+1$ 包含 head_dim 和 LSE。

## 7. 实验和效果

### 7.1 End-to-End Serving

在 SGLang v0.3.4 上，模型包括 Llama 3.1 8B（1xH100）和 70B（4xH100），workload 包括 ShareGPT 和 synthetic variable-length workloads。

摘要级结果：

- 相比 compiler backend，inter-token latency 降低 29-69%。
- 长上下文 inference latency 降低 28-30%。
- parallel generation 加速 13-17%。

这些结果主要说明 FlashInfer 的收益来自 serving 的真实动态性，而不是只在规则 dense attention 微基准上快。

### 7.2 动态序列长度和 Load Balancing

论文测试 constant、uniform、skewed 三种 sequence length 分布。FlashInfer 在 uniform 和 skewed 下更明显，因为 scheduler 会拆分长 KV request 并均衡 CTAs。

附录 ablation 中，长输入分布 $U(4096, 16384)$ 下，不使用 load balancing 的 ITL 明显变差；使用 load balancing 后接近或优于 Triton backend。这个实验验证 scheduler 不是附属功能，而是 serving attention 的核心部分。

### 7.3 Sparse/Dense KV 和 GQA

附录 sparse gathering 实验显示：

- decode 下 sparse KV 与 dense KV 的 bandwidth utilization gap 基本可忽略。
- prefill 下 sparse paged KV 相比 dense KV 有约 10% gap，主要因为 Hopper TMA 更适合固定 stride dense load，sparse gather 需要 async copy 和手动地址计算。

GQA 的 head-group fusion 对短 query length 特别重要，因为多个 query heads 共享同一 KV head，合并后一次 KV load 可以服务多个 query heads。

### 7.4 Shared Prefix / Parallel Generation

Composable formats 在 parallel generation 中把 shared prefix 和 unique suffix 拆开。MLC-Engine 实验中 parallel degree 从 1 到 64，收益在中等 parallel degree 更稳定；n=4 时效果最大，8B 和 70B 都有 ITL/TTFT 改善。

附录 shared-prefix kernel 表明，prefix 很长、batch size 较大时 composable format 比 single format latency 低很多；但真实端到端收益会受 shared prefix 实际长度和非 attention 部分影响。

### 7.5 Fine-Grained Sparsity

在 Quest 这类 long-context KV pruning 场景，FlashInfer 支持小 block size / vector-sparse gather。附录中在 H100 上比较 PyTorch SDPA 和 FlexAttention，FlashInfer 在长序列和小 page budget 下可以达到数量级优势，论文报告最高约 20x。

可信度判断：FlashInfer 的实验覆盖了 kernel microbenchmark、SGLang/vLLM/MLC-Engine 集成和多个 attention 变体，说明它不是单点技巧。但端到端结果仍依赖具体 serving 框架的 host overhead、batching 策略、模型和请求分布。

## 8. 我的判断

真正贡献不是“又写了一个 FlashAttention kernel”，而是提出了 serving attention 的三个工程抽象：

1. KV-cache 访问关系可以统一成 block-sparse matrix。
2. Attention 变体可以通过 JIT functor 注入高性能模板。
3. 动态 sequence lengths 应由 runtime scheduler 生成 plan，再交给固定形态 kernel 执行。

值得精读：

- Section 2.2：attention state composition，这是 split-K 和 composable formats 的数学基础。
- Section 3.1：BSR 如何统一 page table、radix tree、tree attention、sparse mask。
- Section 3.2：template/JIT 如何支持 attention variants。
- Section 3.3：load-balanced scheduler 和 CUDA Graph 兼容。
- Appendix A/B/D/G：GQA head-group fusion、sparse gather overhead、workspace、fine-grained sparsity。

可以快速看：

- Related Work。
- 具体 API 样例，除非要集成 FlashInfer。

价值判断：

- 对算法理解价值：中。它不改变 exact attention 数学，但 attention state composition 很关键。
- 对推理系统价值：高。它把 serving attention 的 KV layout、调度和 CUDA Graph 问题放在同一框架里。
- 对 kernel / 算子优化价值：高。涉及 sparse gather、shared memory packing、GQA fusion、FA2/FA3、CUTLASS 模板和 workspace。
- 是否值得复现：中高。完整复现成本高，但可以先复现 attention state 合并、BSR paged decode 和 load-balanced split-K。

## 9. 下一步阅读建议

建议按这个顺序读：

1. Figure 1：建立 FlashInfer 的 compiler/runtime 总图。
2. Section 2.2：精读 $O(I)$、$\operatorname{LSE}(I)$ 和 $\oplus$，这是整篇论文的数学支点。
3. Figure 2：看 page table 如何变成 BSR。
4. Figure 3：理解 composable formats 为什么能优化 shared prefix。
5. Section 3.2.1-3.2.4：看 sparse gather、tile size、GQA、custom functor。
6. Section 3.3 和 Appendix D：看 plan/run、workspace 和 CUDA Graph。
7. Appendix A/B/G：按需要补 GQA fusion、sparse gather overhead、fine-grained sparsity。

如果要复现，从三个小任务开始：

```text
1. 实现 attention state merge，验证 split 两段 KV 后输出等价于完整 attention。
2. 用 PagedAttention block table 构造一个 BSR indices，写 PyTorch 版 paged decode attention。
3. 写一个简单 scheduler，把长 KV sequence split 成 chunks，再用 attention state merge 合并。
```

最重要的 mental model：

```text
FlashInfer = serving attention 的执行引擎
           = ragged Q/O
           + block-sparse KV access graph
           + composable attention states
           + JIT-customized CUDA/CUTLASS templates
           + runtime load-balanced plan/run scheduler
```
