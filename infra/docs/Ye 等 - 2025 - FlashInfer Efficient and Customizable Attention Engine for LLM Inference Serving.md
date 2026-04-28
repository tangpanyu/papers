# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

论文：Zihao Ye 等，MLSys 2025，arXiv:2501.01005v2

## 一句话总结

FlashInfer 不是只做一个更快的 FlashAttention kernel，而是给 LLM serving 做了一个 attention engine：用统一的 block-sparse 表示兼容各种 KV-cache 管理方式，用可定制 CUDA/CUTLASS 模板和 JIT 生成不同 attention 变体，再用运行时调度器处理 batch 内动态序列长度和负载不均。

## 背景问题

LLM 推理里的 attention 和训练阶段很不一样。训练通常是相对规整的大矩阵计算，而 serving 会遇到：

- prefill、decode、incremental prefill 等不同计算形态；
- 每个请求的 query length 和 KV-cache length 不同；
- PagedAttention、RadixAttention、prefix cache、speculative decoding tree attention 等 KV-cache 组织方式不同；
- GQA/MQA、sliding window、ALiBi、logits soft cap、RoPE 融合、FlashSigmoid 等 attention 变体越来越多；
- CUDA Graph 希望 kernel launch 配置、指针等保持静态，但 serving 的序列长度每步都在变。

论文的核心判断是：如果每个 serving 框架都为自己的 KV-cache 格式和 attention 变体写专用 kernel，维护成本高，也很难覆盖新模型和新 decoding 策略。

## 关键基础

### FlashAttention

FlashAttention 的核心是 online softmax，不显式 materialize attention matrix，而是在片上内存里增量维护输出和 log-sum-exp。FlashInfer 继承 FlashAttention2/3 的主干：Ampere/Ada 上用 FA2 风格，Hopper 上用 FA3 风格。

论文强调了 serving 下的 operational intensity：decode 时 query length 很短，attention 往往更容易受 KV-cache 读取带宽限制。GQA/MQA 通过多个 query heads 共享 KV heads，提高了 KV-cache 复用机会。

### Attention State 可组合

对同一个 query，如果把 key/value 拆成多个片段分别算 attention，只要每个片段保存：

- 局部输出 `O(I)`
- 局部 `LSE(I)`

就能用一个结合、交换的 `⊕` 操作合并成全局 attention 输出。这一点支撑了 FlashInfer 的 split-K、shared-prefix 分解、partial output contraction 等设计。

## 设计一：用 Block-Sparse 统一 KV-cache 表示

FlashInfer 把各种 KV-cache 管理方式抽象成 BSR（Block Sparse Row）矩阵：

- Page table 可以看成 sparse matrix：非零块表示某个 request 的 query tile 访问了哪些物理 KV pages。
- Radix tree / prefix cache / tree attention / KV pruning mask 也可以映射成稀疏访问关系。
- Query 和 output 用 ragged tensor 存，不做 padding。
- KV-cache 用 BSR 存，block size 是 `(Br, Bc)`：
  - `Br` 通常和 query tile size 对齐；
  - `Bc` 由 KV-cache 管理算法决定；
  - FlashInfer 支持任意 `(Br, Bc)`，包括细粒度 vector sparsity。

这和 PagedAttention 的关系是：PagedAttention 解决 KV-cache 内存管理和碎片问题；FlashInfer 把 page table 进一步当作 block-sparse attention 的索引格式来生成高性能 kernel。它不是替代 PagedAttention，而是可以作为 PagedAttention/RadixAttention 等格式上的 attention 计算后端。

## 设计二：Composable Formats 优化 shared prefix

单一 block-sparse 格式有一个 tradeoff：

- `Br` 大：多个 query 可以共享同一批 KV-cache，shared memory/register 复用好；
- `Br` 小：碎片少，灵活，但复用差，更多走 global memory/L2。

FlashInfer 引入 composable formats：同一个 attention 被拆成多个 block-sparse 子格式计算。例如 parallel generation 里，多条候选共享长 prefix，但 suffix 各不相同：

- shared prefix 用较大的 `Br`，让同组 query 共享 KV；
- unique suffix 用较小的 `Br`，避免碎片；
- 不移动 KV-cache 数据，只重新构造 sparse indices 和 indptr。

这个设计特别适合 prefix caching、parallel generation、multi-prefix decoding。

## 设计三：CUDA/CUTLASS 模板加 JIT 支持 attention 变体

FlashInfer 的 compute abstraction 是一套 CUDA/CUTLASS attention 模板：

- dense 和 block-sparse KV-cache 共用主计算骨架；
- sparse KV 先从离散 global memory gather 到连续 shared memory，再用 dense tensor core 计算；
- Hopper 上 contiguous KV 可以用 TMA，sparse gather 因为是非仿射访问，主要用 Ampere-style async copy；
- 支持多个 tile size：query tile size 包括 `1, 16, 32, 64, 128`，KV tile size 包括 `32, 64, 128`，根据 workload 和硬件资源启发式选择。

Attention 变体通过用户定义的 functor 注入模板：

- `QueryTransform`
- `KeyTransform`
- `ValueTransform`
- `OutputTransform`
- `LogitsTransform`
- `LogitsMask`

这使得 FlashInfer 可以生成 fused RoPE、logits soft cap、sliding window、custom mask、非 softmax attention（如 FlashSigmoid）等 kernel。论文里举例说，Streaming-LLM 的 RoPE + attention 融合只需要大约 20 行 query/key transform 代码。

论文选择 CUDA/CUTLASS 而不是 Triton 的原因主要是 Hopper 特性和底层资源控制：CUTLASS 更容易用 warp specialization、TMA、PTX intrinsic，并能做寄存器级调优。

## 设计四：动态负载均衡调度，同时兼容 CUDA Graph

Serving 的 batch 内序列长度变化很大。如果直接按 request 分配 CTA，长请求会拖住部分 SM，短请求对应 SM 很早空闲。

FlashInfer 的 scheduler 做法：

1. 输入每个 request 的 query length 和 KV length，以及 query tile size。
2. 根据总 KV work 和 CTA 数量估计一个最大 KV chunk size。
3. 把长 KV 拆成多个 chunks。
4. 用类似 Stream-K 的 greedy scheduling，把 work chunks 分配给 CTA，目标是各 CTA cost 接近。
5. Attention kernel 产出 partial attention states。
6. Contraction kernel 用 attention composition `⊕` 合并 partial outputs。

为了兼容 CUDA Graph：

- kernel grid size 固定；
- workspace buffer 预先按上界分段分配，指针稳定；
- 每个 generation step 只有 `plan()` 在 CPU 上根据当前 sequence length 更新调度元数据；
- `run()` 可以被 CUDA Graph capture；
- 同一层/同一 generation step 的 plan 可以跨多层复用，摊薄 CPU 开销。

这个 plan/run 拆分类似 inspector-executor 模式：先 inspect irregular workload，再 executor 执行固定形态 kernel。

## 编程接口

用户需要提供：

- attention specification；
- task information；
- workspace buffer；
- sequence length info。

初始化时 JIT compile kernel 并缓存。运行时每步：

1. 更新 sequence length。
2. 调用 `plan()` 生成调度。
3. replay CUDA Graph 执行 `run()`。

Composable formats 会创建多个 attention wrappers，对应不同 block size 或平均 query length；serving 框架在运行时选择合适的 CUDA Graph。

## 实验结论

### End-to-end serving

在 SGLang v0.3.4 上比较 Triton backend，模型包括 Llama 3.1 8B（1xH100）和 70B（4xH100），任务包括 ShareGPT 和 synthetic variable workload。

论文摘要给出的总效果：

- ITL 相比 compiler backend 降低 29-69%；
- 长上下文 inference latency 降低 28-30%；
- parallel generation 加速 13-17%。

### 输入动态性的 kernel 实验

固定 batch size 16，测试 constant、uniform、skewed 三种 sequence length 分布。

FlashInfer 在 uniform 和 skewed 场景明显优于 FlashAttention，原因主要是：

- load-balanced scheduler 缓解长短序列导致的 SM idle；
- decode 使用更合适的小 query tile；
- 对 GQA 有 head-group fusion，提高短 query length 下 KV-cache 复用。

### Streaming-LLM 长上下文

FlashInfer 通过 JIT 生成 fused RoPE + attention kernel。相比 unfused RoPE + attention：

- end-to-end ITL 降低约 28-30%；
- kernel bandwidth utilization 提高约 1.6-3.7x。

这部分说明 JIT customizability 不只是接口优雅，而是能把模型算法里的变体直接转成 fused kernel。

### Parallel generation

在 MLC-Engine 里测试 composable formats。parallel degree `n` 从 1 到 64。

结果：

- 中等 parallel degree（4 到 32）收益稳定；
- n=4 时收益最大：
  - 8B ITL 降低 13.73%，TTFT 降低 16.41%；
  - 70B ITL 降低 17.42%，TTFT 降低 22.86%。
- n 太小，共享 prefix 带来的 block size 增大不足；
- n 太大，整体不再被 attention 主导，收益趋于平台。

### 细粒度稀疏

在 Quest 这类 long-context KV pruning 场景中，FlashInfer 支持 fine-grained block sparsity。附录结果显示，在长序列和小 page budget 下，FlashInfer 相比 PyTorch SDPA / FlexAttention 可达到数量级优势，论文称最高约 20x。

## 和相关工作的区别

- 相比 FlashAttention：FlashInfer 继承其 online-softmax 主计算思想，但重点扩展到 serving 的 sparse KV、变长调度、attention 变体和 CUDA Graph 兼容。
- 相比 PagedAttention：PagedAttention 更偏 KV-cache 内存管理；FlashInfer 把 page table/radix tree 等统一成 BSR，负责高性能 attention 计算。
- 相比 FlexAttention：FlexAttention 接口灵活，基于 Triton/PyTorch compiler；FlashInfer 更偏 serving 和 CUDA/CUTLASS 性能，支持 query/key transform、vector sparsity、load balancing。
- 相比 Hydragen/RelayAttention/ChunkAttention：那些工作专门优化 shared-prefix；FlashInfer 用 composable formats 把 shared-prefix 作为统一 sparse format 的一个特例，便于集成到已有 serving 框架。

## 局限和未来工作

- 当前主要支持 forward attention，不支持训练所需 backward kernel。
- `plan()` 仍在 CPU 上做，虽然可跨层复用，但 host overhead 仍可能影响集成，论文也提到 vLLM bf16 场景有 Python overhead。
- Sparse gather 在 Hopper 上不能直接利用 TMA，dense 和 sparse prefill 之间仍有约 10% gap。
- 未来方向包括把更高层 DSL 编译到 FlashInfer attention spec，以及支持 Triton/其他后端。

## 我的理解

这篇论文的价值不在于某一个单点优化，而在于把 LLM serving 里的 attention 问题重新抽象成三件事：

1. KV-cache 访问关系是一个 sparse matrix。
2. Attention kernel 是一个可参数化模板，模型变体用 functor 注入。
3. 动态序列长度不要写死在 kernel 里，而是交给运行时 scheduler 生成 plan。

这三个抽象组合起来，FlashInfer 就可以同时覆盖 PagedAttention、RadixAttention、prefix cache、tree attention、KV pruning、GQA/MQA、RoPE fusion 和长短序列混合 batch。它更像一个面向 serving 的 attention kernel compiler/runtime，而不是单个 attention 算子库。

如果和你打开的 PagedAttention 笔记连起来看，可以把关系记成：

- PagedAttention：解决 KV-cache 怎么分配、复用、避免碎片。
- FlashInfer：解决在这些复杂 KV-cache 布局上怎么高效算 attention，并且适配不断变化的 attention 变体和请求动态性。
