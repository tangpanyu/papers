# Efficient Memory Management for Large Language Model Serving with PagedAttention

## 论文一句话总结

这篇论文本质上不是改 Transformer 数学，而是把 LLM serving 里的 KV cache 从“每个 request 一大段连续显存”改成“按固定大小 block/page 管理的非连续显存”，再让 attention kernel 通过 block table 间接读取这些 KV。这样 vLLM 可以按需分配、减少碎片、复用 prompt/prefix/beam 的 KV cache，从而用同样显存塞进更大的 continuous batch。

## 1. 背景和问题

LLM 在线服务主要分两段：prefill 对整段 prompt 做并行计算，生成 prompt 的 KV cache；decode 每步只输入一个新 token，但要读取所有历史 K/V 来算 attention。decode 阶段通常 memory-bound，因为每生成一个 token 都要扫描历史 KV cache，而矩阵规模很小，GPU 算力难以吃满。

问题不在模型参数，而在 KV cache 管理。以 OPT-13B 为例，单 token KV cache 大约是：

```text
2(K,V) * 5120(hidden) * 40(layers) * 2 bytes(FP16) = 800 KB/token
```

最大 2048 token 时，一个 request 的 KV cache 可到约 1.6GB。已有系统常为一个 request 预留最大长度的连续 KV tensor，导致 reserved slots、internal fragmentation、external fragmentation，并且难以共享 parallel sampling、beam search、shared prefix 里的重复 KV。

## 2. 核心结论

PagedAttention 证明：LLM serving 的 KV cache 更像操作系统里的虚拟内存，而不是普通 DL 框架里的静态连续 tensor。把 KV cache 分成固定大小 block 后，一个 request 的逻辑连续上下文可以映射到显存中非连续的 physical blocks。

主要收益来自：

- 按需分配 KV block，避免为未知输出长度提前占满最大长度。
- 固定 block size，基本消除外部碎片。
- 最多浪费最后一个 block，显著降低内部碎片。
- 通过引用计数和 copy-on-write 复用 prompt、shared prefix、beam candidate 的 KV block。
- 更大的有效 batch size 提高 decode 吞吐。

代价是 attention kernel 需要查 block table，访存变成间接索引，kernel 有额外分支和非连续访问。论文报告 PagedAttention 的 attention kernel latency 比 FasterTransformer attention kernel 高约 20-26%，但端到端吞吐仍明显更好，因为系统瓶颈是能同时 batch 多少 request。

限制：它主要优化 memory-capacity-bound 的在线生成服务；如果 workload 已经 compute-bound，或者序列很短、显存很宽裕，收益会变小。

## 3. 方法总览

### 标准 attention 参照

标准 causal self-attention 可以写成：

```python
Q: [B, H, S_q, D]
K: [B, H, S_k, D]
V: [B, H, S_k, D]

score = Q @ K.transpose(-1, -2) / sqrt(D)
score = causal_mask(score)
P = softmax(score, dim=-1)
O = P @ V
```

在 serving decode 阶段，通常：

```python
Q_new: [num_active_seq, H, 1, D]
K_cache/V_cache: [num_active_seq, H, S, D]
O_new: [num_active_seq, H, 1, D]
```

每个 active sequence 每步只生成一个 token，但要读完整历史 `S` 个 token 的 K/V。

### baseline 怎么做

FasterTransformer/Orca 这类系统通常让每个 request 的 KV cache 在显存里是一段连续区域，并且为了输出长度不确定，会按最大长度或某种预测长度预留空间：

```text
request A KV: [max_seq_len, layers, heads, head_dim] contiguous
request B KV: [max_seq_len, layers, heads, head_dim] contiguous
```

这带来三个浪费：

- `reserved`：为未来还没生成的 token 占位置。
- `internal fragmentation`：最终序列短于预留长度，尾部永远不用。
- `external fragmentation`：不同大小 chunk 反复申请释放后，显存里出现不可用空洞。

### PagedAttention 改了什么

PagedAttention 把每个 sequence 的 KV cache 切成 logical KV blocks，每个 block 存固定数量 token 的 K/V，例如 block size = 16。logical block 通过 block table 映射到 physical KV block：

```text
sequence logical blocks:
  logical block 0 -> physical block 7
  logical block 1 -> physical block 1
  logical block 2 -> physical block 3

physical blocks 在 GPU DRAM 中不需要连续。
```

attention kernel 不再假设 `K_cache[seq]` 是连续长 tensor，而是：

```text
给定 sequence id 和 token position:
  logical_block_id = position // block_size
  offset           = position % block_size
  physical_block   = block_table[seq][logical_block_id]
  K/V 地址          = kv_pool[physical_block, offset, ...]
```

本质差异：

- 标准 serving KV：每个 request 拥有一段连续大数组。
- PagedAttention KV：所有 request 共享一个 physical block pool，每个 request 只拥有一张 logical-to-physical 映射表。

这不是减少 attention 的理论复杂度。decode 仍然要看历史所有 token，复杂度仍是 `O(S * H * D)`。它减少的是 KV cache 显存浪费和重复存储，从而提高 batch size。

## 4. 关键公式 / 算法

### 4.1 Block-wise attention 公式

**作用**

把原本对连续 `K[1:i], V[1:i]` 的 attention，改写成对多个 KV block 的 attention。数学结果等价于标准 causal attention，只是 K/V 的物理布局变了。

**符号和 shape**

设 block size 为 `B_blk`。第 `j` 个 KV block：

```text
K_j = [k_{(j-1)B_blk+1}, ..., k_{jB_blk}]
V_j = [v_{(j-1)B_blk+1}, ..., v_{jB_blk}]

K_j shape: [B_blk, D]       # 单 head 单 layer 视角
V_j shape: [B_blk, D]
q_i shape: [D]
A_ij shape: [B_blk]         # query i 对第 j 个 block 内 token 的 attention weights
```

对第 `i` 个 query token，需要访问 `ceil(i / B_blk)` 个 logical blocks。论文把注意力写成 block 形式：

```text
A_ij = softmax(q_i K_j^T / sqrt(D)) 在所有可见 block 拼接后的维度上归一化
o_i  = sum_j A_ij V_j
```

更准确地说，softmax 的分母不是只在单个 block 内归一化，而是在所有历史 block 的所有历史 token 上归一化。block 只是计算和存储单位。

**直觉**

如果把所有 block 按逻辑顺序拼回去：

```python
K_full = concat([K_0, K_1, ..., K_m], dim=0)
V_full = concat([V_0, V_1, ..., V_m], dim=0)
```

那么 PagedAttention 的输出应与：

```python
p = softmax(q @ K_full.T / sqrt(D))
o = p @ V_full
```

完全一致。PagedAttention 只是让 `K_full/V_full` 不必物理连续。

**PyTorch 语义**

```python
def paged_attention_one_seq(q, kv_pool_k, kv_pool_v, block_table, seq_len, block_size):
    # q: [H, D]
    # kv_pool_k/v: [num_physical_blocks, H, block_size, D]
    # block_table: [num_logical_blocks] -> physical block id
    keys = []
    vals = []
    for logical_bid in range((seq_len + block_size - 1) // block_size):
        physical_bid = block_table[logical_bid]
        start = logical_bid * block_size
        valid = min(block_size, seq_len - start)
        keys.append(kv_pool_k[physical_bid, :, :valid, :])  # [H, valid, D]
        vals.append(kv_pool_v[physical_bid, :, :valid, :])  # [H, valid, D]

    K = torch.cat(keys, dim=1)  # [H, S, D]
    V = torch.cat(vals, dim=1)  # [H, S, D]
    score = torch.einsum("hd,hsd->hs", q, K) / math.sqrt(q.shape[-1])
    prob = torch.softmax(score, dim=-1)
    out = torch.einsum("hs,hsd->hd", prob, V)
    return out
```

真实 kernel 不会真的 `cat`，而是边查 block table 边加载 physical block，并在 kernel 内完成 score、softmax、value accumulation。

### 4.2 Block table 与按需分配

**作用**

让一个 request 的逻辑 KV 序列保持连续，但 physical KV blocks 可以散落在显存池里。

**关键状态**

```text
block_table[request_id][logical_block_id] = physical_block_id
num_filled[request_id][logical_block_id] = 这个 logical block 已写入多少 token
ref_count[physical_block_id] = 被多少 logical block 引用
```

对每个 layer/head，可以有独立 block table，也可以把所有 layer/head 的 K/V 作为一个大 block 管理。论文实现选择了前者，主要是工程简单；两种设计性能本质相同。

**append token 语义**

```python
def append_token(seq, new_k, new_v):
    logical_bid = seq.length // block_size
    offset = seq.length % block_size

    if offset == 0:
        physical_bid = allocate_free_physical_block()
        seq.block_table.append(physical_bid)
        ref_count[physical_bid] = 1
    else:
        physical_bid = seq.block_table[logical_bid]
        if ref_count[physical_bid] > 1:
            # copy-on-write
            new_physical_bid = allocate_free_physical_block()
            copy_block(src=physical_bid, dst=new_physical_bid)
            ref_count[physical_bid] -= 1
            ref_count[new_physical_bid] = 1
            seq.block_table[logical_bid] = new_physical_bid
            physical_bid = new_physical_bid

    kv_pool_k[physical_bid, :, offset, :] = new_k
    kv_pool_v[physical_bid, :, offset, :] = new_v
    seq.length += 1
```

**影响**

一个 sequence 只有最后一个 block 可能没填满，所以每个 request 的浪费上界约为一个 block。block size 越小，碎片越少，但 kernel 并行度和访存效率可能变差；block size 越大，kernel 更顺，但内部碎片和共享粒度变差。论文最终默认 block size = 16。

### 4.3 Copy-on-write 共享 KV

**作用**

支持 parallel sampling、beam search、shared prefix 的 KV cache 共享。

例如 parallel sampling 中，一个 prompt 生成多个 sample。prompt 部分 KV 完全相同，可以让多个 sequence 的 logical blocks 指向同一组 physical blocks：

```text
sample A logical block 0 -> physical block 7
sample B logical block 0 -> physical block 7
ref_count[7] = 2
```

一旦某个 sample 需要往共享的最后一个 block 写新 token，如果 `ref_count > 1`，就复制这个 block，再写入自己的新 token。这就是 block 粒度的 copy-on-write。

**beam search**

beam search 的共享更强。多个 beam candidates 可能共享 prompt 和早期生成路径，后来才分叉。传统系统可能需要反复复制大段 KV cache；vLLM 只复制需要写入的最后一个共享 block，其余历史 block 继续共享。

**shared prefix**

对系统 prompt、few-shot examples 这类很多请求共享的 prefix，服务端可以预先计算并保留 prefix 的 physical KV blocks。后续请求的 logical blocks 直接映射到这些 cached physical blocks，只对用户输入部分做 prefill。

## 5. 数据流和实现设计

### 5.1 Prefill

输入：

```text
prompt token ids: [S_prompt]
```

常规 attention 或 FlashAttention 可以一次并行处理整个 prompt：

```text
hidden: [S_prompt, hidden_dim]
Q/K/V:  [layers, H, S_prompt, D]
```

vLLM 在 prefill 后把生成的 K/V 写入 KV block pool：

```text
logical block 0: token 0..15
logical block 1: token 16..31
...
last block: 可能未满
```

如果 prompt 长度是 7，block size 是 4，那么需要 2 个 logical blocks。最后一个 block 只填了 3 个 token，剩下 1 个 slot 留给后续 decode token。

### 5.2 Decode

每个 decode iteration：

1. scheduler 选择一批 active sequences。
2. KV cache manager 为本轮会新增 token 的 sequence 分配必要 physical blocks。
3. scheduler 把 input token ids 和每个 sequence 的 block table 发给 GPU workers。
4. GPU worker 执行模型。
5. attention layer 通过 PagedAttention kernel 按 block table 读取历史 K/V。
6. 当前 token 的新 K/V 被写入对应 physical block。
7. 采样得到 next token，返回 scheduler。

decode attention 的单 sequence 单 layer 单 head 数据流：

```text
q_new [D]
  -> 查 block_table: logical block ids -> physical block ids
  -> 从 kv_pool 逐 block 读 K [block_size, D]
  -> 算 score [block_size]
  -> 跨所有 block 做 softmax 归一化
  -> 逐 block 读 V [block_size, D]
  -> 累加输出 o [D]
```

真实 kernel 会融合 block read 和 attention，避免把非连续 blocks 先 gather 成连续 K/V。

### 5.3 KV cache layout

概念 layout：

```text
K pool: [num_physical_blocks, num_layers, num_kv_heads, block_size, head_dim]
V pool: [num_physical_blocks, num_layers, num_kv_heads, block_size, head_dim]
```

论文实现里也提到可以按 layer/head 拆成不同 block table 方便实现。多 GPU tensor parallel 时，每个 GPU worker 拥有同样的 physical block ids 和 block table 语义，但只存自己负责的 attention heads 的 KV cache。

### 5.4 Continuous batching

PagedAttention 本身不是 batching 策略，但它让 iteration-level scheduling 更有效。传统 Orca 的 iteration-level scheduling 可以在每步加入/移除请求，但如果 KV cache 预留和碎片严重，显存仍然会限制 batch size。

vLLM 的效果来自两者配合：

```text
iteration-level scheduler 负责每步动态组 batch
PagedAttention/KV manager 负责让更多 active sequences 的 KV 放进显存
```

所以它提升吞吐的直接路径是：

```text
减少 KV 浪费 -> 可容纳更多 active request/sequence -> decode batch size 更大 -> 权重读取和 kernel launch 成本被摊薄 -> throughput 更高
```

### 5.5 调度和 preemption

vLLM 使用 FCFS。请求过载或 physical blocks 不够时，需要 preempt 一些 sequence。论文强调 LLM serving 的特殊性：一个 sequence decode 时通常要访问它全部历史 KV，所以 eviction 采用 all-or-nothing，即要么保留某个 sequence 的所有 blocks，要么整体驱逐。

恢复有两种：

- swapping：把 evicted KV blocks 拷到 CPU RAM，需要时拷回 GPU。
- recomputation：丢掉 KV，恢复时把已生成 token 和原 prompt 拼成新 prompt，重新 prefill 出整段 KV。

小 block size 下 swapping 会产生很多小 PCIe copy，效率差；recomputation 不依赖 block size。论文实验显示 block size 16-64 时两者端到端性能接近。

### 5.6 Kernel 设计

PagedAttention 需要几个关键 kernel：

- fused reshape and block write：把新 K/V reshape 成 block-friendly layout，并按 block table 写入 physical block。
- fused block read and attention：attention kernel 内部查 block table，读取 non-contiguous KV blocks 并计算 attention。
- fused block copy：copy-on-write 可能需要复制多个不连续 block，用一个 kernel 批量完成，避免大量小 `cudaMemcpyAsync` launch。

kernel 代价：

- 额外读取 block table。
- 更多分支，尤其处理 variable sequence length 和最后一个未满 block。
- physical blocks 非连续，访存局部性比连续 KV tensor 差。

但 decode 的系统瓶颈往往不是单个 attention kernel latency，而是显存容量限制导致 batch size 上不去。

## 6. 实验和效果

### 6.1 设置

模型：

- OPT-13B：1 张 A100 40GB，参数约 26GB，KV cache 可用约 12GB。
- OPT-66B：4 张 A100，总 160GB，参数约 132GB，KV cache 可用约 21GB。
- OPT-175B：8 张 A100 80GB，总 640GB，参数约 346GB，KV cache 可用约 264GB。
- 另有 LLaMA-13B 用于 shared prefix 翻译实验。

workload：

- ShareGPT：prompt 和 output 更长，均值约 input 161 tokens、output 338 tokens。
- Alpaca：更短，均值约 input 19 tokens、output 58 tokens。

baseline：

- FasterTransformer：低延迟优化强，但缺少 Orca 式细粒度调度。
- Orca(Max)：按最大长度预留，最保守。
- Orca(Pow2)：按 2 的幂近似预留。
- Orca(Oracle)：假设提前知道真实输出长度，是不现实的上界 baseline。

指标：normalized latency，即 end-to-end latency / output length。系统能承受的 request rate 越高，吞吐越好。

### 6.2 Basic sampling

在单输出 sampling 中，vLLM 相比 Orca(Oracle) 在 ShareGPT 上可承受约 1.7-2.7 倍 request rate，相比 Orca(Max) 约 2.7-8 倍。相比 FasterTransformer，最高可达 22 倍 request rate，因为 FasterTransformer 既没有细粒度 scheduling，也有低效 KV 管理。

关键解释：ShareGPT 序列更长，KV cache 成为更强瓶颈，因此 PagedAttention 的显存利用率收益更明显。

在 Alpaca 上趋势类似，但 OPT-175B 场景里 vLLM 相对 Orca(Oracle/Pow2) 优势变小。原因是 175B 配置有很大的 KV cache 空间，而 Alpaca 序列短，此时系统更接近 compute-bound，memory management 不再是唯一瓶颈。

### 6.3 Parallel sampling 和 beam search

parallel sampling 中，多个 samples 共享 prompt KV。vLLM 随 parallel size 增大收益更高。论文报告 Alpaca 上 KV block sharing 节省：

```text
parallel sampling: 6.1% - 9.8%
beam search:       37.6% - 55.2%
```

ShareGPT 上因为 prompt 更长，共享收益更大：

```text
parallel sampling: 16.2% - 30.5%
beam search:       44.3% - 66.3%
```

beam search 的收益更高，因为 beam candidates 不只共享 prompt，还能共享早期生成路径。OPT-13B Alpaca 上，vLLM 相比 Orca(Oracle) 的提升从 basic sampling 的约 1.3 倍，提高到 beam width = 6 时的约 2.3 倍。

### 6.4 Shared prefix

翻译任务中共享 prefix：

- 1-shot prefix，约 80 token：vLLM 比 Orca(Oracle) 吞吐高 1.67 倍。
- 5-shot prefix，约 341 token：vLLM 比 Orca(Oracle) 吞吐高 3.58 倍。

prefix 越长，预计算和共享 KV 的价值越大。

### 6.5 Chatbot

chatbot 把历史对话和最后一个用户 query 拼成 prompt。实验中 prompt 截断到最近 1024 tokens，最多生成 1024 tokens。ShareGPT 里很多长对话会让 Orca baseline 为 output 也预留大量空间，三种 Orca baseline 表现接近。vLLM 因为按需分配，能承受约 2 倍 request rate。

### 6.6 Ablation

attention kernel 微基准中，PagedAttention kernel 比 FasterTransformer attention kernel 慢约 20-26%。这是 block table 和非连续访问的代价。

block size：

- 太小：每次读的 block 太碎，GPU 并行度和访存效率差。
- 太大：内部碎片增加，共享粒度变粗。
- 论文实践默认 block size = 16。

## 7. 我的判断

真正贡献不是“提出一种新的 attention 数学公式”，而是把 LLM serving 的主要资源瓶颈明确建模为 KV cache virtual memory problem，并且把 OS 里的 paging、block table、reference count、copy-on-write 改造成 GPU serving 系统可用的机制。

值得精读：

- Section 3：为什么 KV cache 是 serving bottleneck，以及三类浪费。
- Section 4.1-4.3：PagedAttention、block table、decode append 流程。
- Section 4.4：parallel sampling、beam search、shared prefix 如何共享 KV。
- Section 5.1：为什么必须改 kernel，而不是只写一个 Python allocator。
- Section 7.1-7.2：indirection overhead 和 block size tradeoff。

可以快速看：

- Introduction 的应用背景。
- Related Work。
- 具体 API/frontend 实现细节。

价值判断：

- 对算法理解价值：中。attention 数学本身没变，但它改变了 serving attention 的内存抽象。
- 对推理系统价值：高。这是 vLLM 的核心系统设计。
- 对 kernel / 算子优化价值：中高。重点不是新算子数学，而是 paged KV layout 下的 attention kernel 访存。
- 是否值得复现：高。最小复现可以不做完整 vLLM，只实现 block table + paged KV cache + decode attention 语义。

## 8. 下一步阅读建议

如果要真正掌握这篇论文，建议按这个顺序：

1. 先看 Figure 3：确认旧系统的 reserved/internal/external fragmentation。
2. 再看 Figure 6：理解 logical block 到 physical block 的映射和 append 过程。
3. 精读 Section 4.3：这是单 sequence decode 的核心流程。
4. 精读 Figure 8/9：parallel sampling 和 beam search 的 copy-on-write 共享。
5. 看 Section 5.1：理解为什么需要 fused block write、fused block read attention、fused block copy。
6. 看 Figure 18：block table 间接寻址确实有 kernel overhead，但系统收益盖过它。

如果复现，建议从最小 PyTorch 语义开始：

```text
1. 实现 physical KV block pool。
2. 实现 per-sequence block table。
3. 实现 append_token，支持最后 block 写入和新 block 分配。
4. 实现 paged_attention_one_seq，先用 Python loop + torch.cat 验证数值等价。
5. 加 batch：不同 sequence 的 block table 长度不同。
6. 加 ref_count + copy-on-write，复现 parallel sampling 的 prompt sharing。
7. 最后再考虑 Triton/CUDA kernel，避免真的 gather/cat。
```

最重要的 mental model：

```text
PagedAttention = 标准 attention 数学
               + KV cache 物理存储改成 page/block pool
               + 每个 sequence 用 block table 维护逻辑连续性
               + scheduler/allocator 按需分配和共享 block
               + attention kernel 支持间接索引读取 K/V
```
