# Efficient Memory Management for Large Language Model Serving with PagedAttention

论文：Woosuk Kwon 等，SOSP 2023，arXiv:2309.06180

## 论文一句话总结

PagedAttention 的本质不是发明新的 attention 数学，而是把 LLM serving 里不断增长的 KV cache 当成“虚拟内存”管理：每个 request 看到逻辑连续的 token KV，GPU 显存里实际存放为离散的 fixed-size physical blocks。

它让 vLLM 用 block table、按需分配、引用计数和 copy-on-write 减少 KV cache 碎片与重复存储，从而在同样显存下放进更大的 continuous batch。

## 1. 背景和问题

LLM 在线推理分两段：prefill 读完整 prompt 并生成 prompt KV cache；decode 每次只输入一个新 token，但 attention 仍要读完整历史 K/V。decode 阶段通常 memory-bound，因为每步的矩阵很小，GPU 算力吃不满，瓶颈变成读取模型权重和 KV cache。

真正棘手的是 KV cache 内存管理。以 OPT-13B 为例，单 token KV cache 大约是 $2 \times 5120 \times 40 \times 2 \approx 800\text{KB}$，最大 2048 token 时一个 request 可占约 1.6GB。传统系统为每个 request 预留连续 KV tensor，输出长度又事先未知，于是产生 reserved slots、internal fragmentation、external fragmentation，并且 parallel sampling、beam search、shared prefix 里的重复 KV 难以共享。

## 2. 核心结论

PagedAttention 证明：KV cache 更像操作系统里的分页内存，而不是普通深度学习框架里的静态连续 tensor。只要 attention kernel 能通过 block table 间接读取 K/V，数学结果仍等价于标准 causal attention，但显存可以按 block 动态分配和共享。

主要提升来自：

- 按需分配 physical KV blocks，避免为未知输出长度提前占满最大长度。
- 所有 physical blocks 等大，基本消除外部碎片。
- 每个 sequence 最多浪费最后一个未填满 block，内部碎片被限制在一个 block 内。
- prompt、shared prefix、beam candidate 可以用引用计数共享 blocks。
- 更高显存利用率带来更大的 decode batch size，进而提高吞吐。

代价是 attention kernel 多了 block table lookup、非连续访存和最后 block 边界处理。论文微基准里 PagedAttention kernel 比 FasterTransformer attention kernel 慢约 20-26%，但端到端吞吐更好，因为系统瓶颈通常是“显存能容纳多少 active sequences”。

限制：它优化的是 memory-capacity-bound 的在线生成服务。如果序列很短、显存富余，或者 workload 已接近 compute-bound，收益会下降。

## 3. 形象解释 / Mental Model

可以把每个 request 的 KV cache 想成一本不断变厚的笔记本。传统 serving 系统会先给每个 request 发一本很厚的空白本，哪怕最后只写几页，整本也一直占着桌面；PagedAttention 则把笔记本拆成固定页，每写满一页才向公共页池拿下一页。

baseline 像这样：

```text
request A: 一整段连续 KV tensor，按 max_seq_len 预留
request B: 一整段连续 KV tensor，按 max_seq_len 预留
```

PagedAttention 像这样：

```text
request A logical blocks: 0 -> physical 7, 1 -> physical 1, 2 -> physical 9
request B logical blocks: 0 -> physical 4, 1 -> physical 2
```

生成下一个 token 时，真实数据流是：

1. scheduler 选出 active sequences。
2. 每个 sequence 的新 query $q_t$ 进入 attention layer。
3. kernel 根据 token position 算出 logical block id 和 offset。
4. kernel 查 `block_table[seq_id][logical_block_id]` 找到 physical block。
5. 从 `kv_pool` 逐 block 读取历史 K/V，完成 softmax attention。
6. 当前 token 的新 K/V 写入最后一个 block；如果最后 block 满了，就分配新 physical block。

如果只记住一张图，应该记住 Figure 6/7 那类图：logical KV blocks 连续，physical KV blocks 离散，中间靠 block table 映射。它说明论文真正移动的瓶颈不是 attention 计算复杂度，而是 KV cache 的显存浪费、共享粒度和调度容量。

## 4. 方法总览

标准 causal attention 参照：

```python
Q: [B, H, S_q, D]
K: [B, H, S_k, D]
V: [B, H, S_k, D]

score = Q @ K.transpose(-1, -2) / sqrt(D)
score = causal_mask(score)
P = softmax(score, dim=-1)
O = P @ V
```

decode 阶段通常是：

```python
Q_new: [num_active_seq, H, 1, D]
K_cache/V_cache: [num_active_seq, H, S, D]
O_new: [num_active_seq, H, 1, D]
```

PagedAttention 不改变 $QK^\top$、softmax、$PV$ 的数学定义。它改变的是 `K_cache/V_cache` 的物理布局。

baseline 连续 KV cache：

```text
K_cache[request_id, layer, head, position, dim]
V_cache[request_id, layer, head, position, dim]
```

PagedAttention KV cache：

```text
K_pool[physical_block_id, layer, kv_head, block_offset, head_dim]
V_pool[physical_block_id, layer, kv_head, block_offset, head_dim]
block_table[sequence_id, logical_block_id] = physical_block_id
```

对任意 token position：

```text
logical_block_id = position // block_size
block_offset     = position % block_size
physical_block   = block_table[sequence_id, logical_block_id]
```

复杂度没有从 $O(SHD)$ 变成 $O(1)$。decode 仍要读历史所有 K/V。收益来自 memory management：

```text
减少 KV 浪费
  -> 同样显存容纳更多 active sequences
  -> decode batch size 更大
  -> 模型权重读取、kernel launch、GPU 执行被更多请求摊薄
  -> serving throughput 提高
```

## 5. 关键公式 / 算法

### 5.1 Block-wise Attention

作用：把连续 K/V 上的标准 attention，改写成对多个 KV blocks 的等价 attention。block 只是存储和 kernel 访问单位，不是新的归一化边界。

设 block size 为 $B_{\text{blk}}$，单 layer、单 head 视角下：

$$
K_j = [k_{jB_{\text{blk}}}, \ldots, k_{(j+1)B_{\text{blk}}-1}] \in \mathbb{R}^{B_{\text{blk}} \times D}
$$

$$
V_j = [v_{jB_{\text{blk}}}, \ldots, v_{(j+1)B_{\text{blk}}-1}] \in \mathbb{R}^{B_{\text{blk}} \times D}
$$

对 query $q_i \in \mathbb{R}^{D}$，令可见的历史 block 集合为 $\mathcal{B}(i)$。实际输出仍是：

$$
o_i =
\sum_{j \in \mathcal{B}(i)}
\sum_{t \in j}
\frac{\exp(q_i^\top k_t / \sqrt{D})}
{\sum_{j' \in \mathcal{B}(i)} \sum_{t' \in j'} \exp(q_i^\top k_{t'} / \sqrt{D})}
v_t
$$

注意 softmax 分母跨所有历史 blocks，而不是每个 block 内单独归一化。直觉上，它等价于先把所有 logical blocks 按顺序拼回完整 $K_{\text{full}}, V_{\text{full}}$，再执行标准 attention。

PyTorch 语义：

```python
def paged_attention_one_seq(q, k_pool, v_pool, block_table, seq_len, block_size):
    # q: [H, D]
    # k_pool/v_pool: [num_physical_blocks, H, block_size, D]
    keys, vals = [], []
    for logical_bid in range((seq_len + block_size - 1) // block_size):
        physical_bid = block_table[logical_bid]
        start = logical_bid * block_size
        valid = min(block_size, seq_len - start)
        keys.append(k_pool[physical_bid, :, :valid, :])
        vals.append(v_pool[physical_bid, :, :valid, :])

    K = torch.cat(keys, dim=1)  # [H, S, D]
    V = torch.cat(vals, dim=1)  # [H, S, D]
    score = torch.einsum("hd,hsd->hs", q, K) / math.sqrt(q.shape[-1])
    prob = torch.softmax(score, dim=-1)
    return torch.einsum("hs,hsd->hd", prob, V)
```

真实 CUDA kernel 不会真的 `cat`，而是在 kernel 内查 block table、加载 physical blocks、做 online softmax 和 value accumulation。

### 5.2 Block Table 与 Append Token

作用：让 sequence 的逻辑上下文连续，同时让物理显存可以非连续分配。

关键状态：

```text
block_table[sequence_id][logical_block_id] = physical_block_id
ref_count[physical_block_id] = 被多少 logical blocks 引用
free_block_queue = 可复用 physical blocks
```

append 一个 token 的语义：

```python
def append_token(seq, new_k, new_v):
    logical_bid = seq.length // block_size
    offset = seq.length % block_size

    if offset == 0:
        physical_bid = allocate_block()
        seq.block_table.append(physical_bid)
        ref_count[physical_bid] = 1
    else:
        physical_bid = seq.block_table[logical_bid]
        if ref_count[physical_bid] > 1:
            new_bid = allocate_block()
            copy_block(src=physical_bid, dst=new_bid)
            ref_count[physical_bid] -= 1
            ref_count[new_bid] = 1
            seq.block_table[logical_bid] = new_bid
            physical_bid = new_bid

    k_pool[physical_bid, :, offset, :] = new_k
    v_pool[physical_bid, :, offset, :] = new_v
    seq.length += 1
```

小例子：block size = 4，prompt 长度 = 7，则 prefill 后有两个 logical blocks。block 0 满，block 1 写了 3 个 token。下一步 decode 会把新 token KV 写入 block 1 的 offset 3；再下一步则需要分配新的 block 2。

影响：block size 越小，内部碎片越小、共享粒度越细，但 block table 更长、kernel 间接访问更多；block size 越大，访存更规整，但最后 block 浪费和 copy-on-write 粒度变粗。论文默认使用 block size 16。

### 5.3 Copy-on-Write KV Sharing

作用：让多个 sequences 共享相同 prompt/prefix 的 KV cache，只有分叉后才复制需要写入的 block。

parallel sampling 中，一个 prompt 生成多个 samples：

```text
sample A block 0 -> physical 10
sample B block 0 -> physical 10
ref_count[10] = 2
```

如果 sample A 要写入一个仍被共享的最后 block，就先 copy-on-write：

```text
copy physical 10 -> physical 23
sample A block 0 -> physical 23
sample B block 0 -> physical 10
```

beam search 的共享更强，因为 beam candidates 不只共享 prompt，还共享早期生成路径。shared prefix 服务场景也类似，系统 prompt 或 few-shot examples 可以预先计算 KV blocks，后续请求直接引用。

## 6. 数据流和实现设计

### 6.1 Prefill

prefill 输入是一段 prompt：

```text
tokens: [S_prompt]
hidden: [S_prompt, hidden_dim]
Q/K/V per layer: [H, S_prompt, D]
```

prefill attention 可以像常规 FlashAttention 一样并行处理整段 prompt。不同之处在于：计算出的 K/V 不写入一个连续 request tensor，而是按 block size 写入 physical block pool。

```text
logical block 0: token 0..15
logical block 1: token 16..31
...
last block: 可能未满，后续 decode 继续写
```

### 6.2 Decode

每个 decode iteration：

1. scheduler 选出本轮 active sequences。
2. KV cache manager 检查这些 sequences 是否需要新 physical blocks。
3. 模型前向计算当前 token 的 $q_t, k_t, v_t$。
4. PagedAttention kernel 根据 block table 读取历史 K/V，输出 attention result。
5. 当前 token 的新 K/V 写入对应 physical block。
6. 采样得到 next token，完成或继续下一轮。

单 sequence、单 layer、单 head 的核心数据流：

```text
q_t [D]
  -> block_table: logical block -> physical block
  -> 逐 block 读 K [block_size, D]
  -> 计算 score [block_size]
  -> 跨所有历史 token 做 online softmax
  -> 逐 block 读 V [block_size, D]
  -> 累加输出 o_t [D]
```

### 6.3 Kernel 需要改什么

PagedAttention 至少需要这些 kernel / kernel 逻辑：

- fused reshape and block write：把新 K/V reshape 到 block-friendly layout，并写入 physical block。
- paged attention：查 block table，从非连续 physical blocks 读取 K/V 并计算 attention。
- fused block copy：copy-on-write 时批量复制 blocks，避免很多小 `cudaMemcpyAsync`。

额外开销来自 block table 读取、非连续 global memory access、边界处理和 variable sequence length。但这些开销换来更大的 batch capacity。

### 6.4 Continuous Batching 与 Preemption

PagedAttention 本身不是调度算法，但它让 iteration-level scheduling 更有效。Orca 式调度可以每轮加入/移除请求；PagedAttention 让这些请求的 KV cache 以更少浪费同时留在 GPU 上。

显存不足时，vLLM 使用 all-or-nothing 的 request/sequence 级 preemption：一个 sequence 要么所有 blocks 都在 GPU，要么整体驱逐，因为 decode attention 需要完整历史 KV。

恢复方式有两类：

- swapping：把 KV blocks 换到 CPU RAM，需要时拷回 GPU。
- recomputation：丢弃 KV，恢复时用 prompt + 已生成 tokens 重新 prefill。

论文实验显示 block size 16-64 时两者端到端性能接近；小 block size 下 swapping 会产生大量小 PCIe copy，recomputation 反而更稳。

## 7. 实验和效果

论文在 OPT-13B、OPT-66B、OPT-175B，以及 LLaMA-13B shared-prefix 场景上评估。workload 包括 ShareGPT、Alpaca、parallel sampling、beam search、translation shared prefix 和 chatbot。

核心结果：

- vLLM 相比 FasterTransformer 和 Orca 通常有 2-4 倍吞吐提升。
- 在 ShareGPT 这类长 prompt/长 output 场景，优势更明显，因为 KV cache 更容易成为显存瓶颈。
- 在 basic sampling 中，vLLM 相比 Orca(Oracle) 仍可承受约 1.7-2.7 倍 request rate；相比 Orca(Max) 可达约 2.7-8 倍。
- FasterTransformer 在一些设置下差距更大，因为缺少细粒度调度和高效 KV 管理。

parallel sampling 和 beam search 的关键不是“attention 算得更快”，而是 KV blocks 能共享。论文报告 KV sharing 节省：

```text
Alpaca parallel sampling: 6.1% - 9.8%
Alpaca beam search:       37.6% - 55.2%
ShareGPT parallel sampling: 16.2% - 30.5%
ShareGPT beam search:       44.3% - 66.3%
```

shared prefix 翻译任务中，prefix 越长收益越大：1-shot prefix 约 80 tokens 时吞吐提升 1.67 倍；5-shot prefix 约 341 tokens 时提升 3.58 倍。

需要注意的是，PagedAttention kernel 单独看更慢。论文微基准显示它比连续 KV attention kernel 慢约 20-26%。这恰好说明论文贡献在系统层：牺牲一点单 kernel latency，换来显存利用率和 batch size 的巨大改善。

## 8. 我的判断

真正贡献：把 LLM serving 的核心资源问题从“怎么写更快 attention”提升为“怎么虚拟化 KV cache”。这套抽象后来成为 vLLM、Paged KV cache、prefix caching、continuous batching 系统的基础概念。

值得精读：

- Section 3：KV cache 三类浪费和为什么 serving memory-bound。
- Section 4.1-4.3：PagedAttention、block table、append 过程。
- Section 4.4：parallel sampling、beam search、shared prefix 的共享。
- Section 5.1：为什么需要专门 kernel。
- Section 7.1-7.3：kernel overhead、block size、swapping vs recomputation。

可以快速看：

- Introduction 的应用背景。
- Related Work。
- API/frontend 细节。

价值判断：

- 对算法理解价值：中。attention 数学没变，但 serving attention 的内存视角很重要。
- 对推理系统价值：高。它是 vLLM 的核心系统设计。
- 对 kernel / 算子优化价值：中高。重点是 paged layout 下的访存与 fused kernels。
- 是否值得复现：高。可以先做 PyTorch 语义版，再考虑 Triton/CUDA。

## 9. 下一步阅读建议

建议按这个顺序读：

1. Figure 3：看清 reserved、internal fragmentation、external fragmentation。
2. Figure 6/7：理解 logical blocks 到 physical blocks 的映射。
3. Section 4.3：精读 decode append 和 block table 访问。
4. Figure 8/9：理解 parallel sampling 和 beam search 的 copy-on-write。
5. Section 5.1：理解为什么必须改 attention kernel。
6. Figure 18/19：看 block size、indirection overhead、swapping/recomputation tradeoff。

如果复现，从最小版本开始：

```text
1. 实现 physical KV block pool。
2. 实现 per-sequence block table。
3. 实现 append_token 和新 block 分配。
4. 实现 paged_attention_one_seq，用 torch.cat 验证数值等价。
5. 支持 batch 内不同 seq_len。
6. 加 ref_count + copy-on-write。
7. 最后再写 Triton/CUDA kernel，避免真正 gather/cat。
```

最重要的 mental model：

```text
PagedAttention = 标准 attention 数学
               + KV cache physical block pool
               + per-sequence logical-to-physical block table
               + on-demand allocation
               + ref_count/copy-on-write sharing
               + attention kernel 间接读取 K/V
```
