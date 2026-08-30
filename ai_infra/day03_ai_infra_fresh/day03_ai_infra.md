# AI Infra Daily --- Day 3

## `tcgen05.cp`：SMEM→TMEM + vLLM `slot_mapping` 真正落到 KV Cache

**总预算：约 55 分钟**：Blackwell/PTX 22 分钟；KV Cache 22 分钟；面试 11
分钟。

今天不扩支线。前两天已经建立 TMEM 地址/`ld` 语义，以及
`BlockPool → block_table → slot_mapping`。今天各往前走一步：Blackwell 看
**什么情况下数据从 SMEM 直接进 TMEM**；vLLM 看 **`slot_mapping`
被谁消费、怎样变成真实 KV cache 写地址**。

# Part 1 --- Blackwell / PTX：`tcgen05.cp`

## 进入主题前：先知道这段在干什么（2 分钟）

前两天你看到 TMEM 主要承担 accumulator；但 `tcgen05` 还提供 SMEM→TMEM
的专用数据路径。`tcgen05.cp` 不是普通 `ld/st` 的替代品：它按 Tensor Core
所需矩阵形状组织搬运，并可在特定低比特格式下顺便做 decompression。

``` text
GMEM --TMA--> SMEM --tcgen05.cp--> TMEM --> tcgen05 consumer
```

进入 PTX
只盯：`SMEM descriptor`、`taddr`、`shape`、`cta_group`、`decompression`。今天不展开低比特编码。

## PTX 导航（约 10 分钟）

官方 PTX ISA：<https://docs.nvidia.com/cuda/parallel-thread-execution/>

先搜索 `Tensor Memory Data Movement Instructions`。PTX 在这里把
`tcgen05.cp` 定义为 shared memory 到 Tensor Memory 的异步 copy。紧接着扫
`Optional Decompression`：4-bit/6-bit custom floating types 可以在 copy
过程中展开到 8-bit。

``` text
cp.async.bulk.tensor / TMA : GMEM → SMEM
tcgen05.cp                 : SMEM → TMEM
```

然后搜索具体 `tcgen05.cp` 指令正文。只抓
`taddr`、`s-desc`、`shape`、`cta_group`。大量 shape
表先不背。你要形成的结论是：destination 用 TMEM address 描述；source
是描述 Tensor Core 所需 SMEM tile 的 descriptor，所以它天然不是
byte-copy。

最后只看 optional decompression 的第一张 4-bit→8-bit 官方示意图即可。

**明确跳过**：所有 4/6-bit bit-level encoding、block scaling、MMA
instruction descriptor、CTA-pair shape 细节、完整 mbarrier 协议。

## CUTLASS 官方数据流（4 分钟）

<https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/tcgen05_programming.html>

先看 `Global Memory (GMEM) to MMA data flow overview`。只追：

``` text
A/B : GMEM → SMEM → MMA
C/D :          TMEM → RMEM → GMEM
```

注意：dense F16/BF16 常规路径里 A/B 可以直接以 SMEM descriptor 给
MMA。因此学了 `tcgen05.cp` 不等于每个 Blackwell GEMM 都必须
SMEM→TMEM；是否使用取决于 operand source / MMA kind / low-bit 等设计。

## CuTe 对照（约 6 分钟）

进入前先知道：PTX 给的是 `taddr + s-desc + shape`；CuTe 负责把"哪个 SMEM
tensor tile 对应哪个 TMEM tensor tile"编码成 `CopyAtom/TiledCopy`。

API：<https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html>

只搜索并读： - `make_s2t_copy(...)` - `get_s2t_smem_desc_tensor(...)` -
遇到 broadcast 时扫 `append_s2t_broadcast_mode(...)`

``` text
SMEM Tensor + TMEM Tensor + S2T CopyAtom
                    ↓
                TiledCopy
                    ↓
                tcgen05.cp
```

最容易误解：CuTe layout 描述 collective copy 如何覆盖逻辑
tensor，**layout 本身不是一次数据搬运**。

# Part 2 --- vLLM：`slot_mapping` 怎么真正写进 KV Cache

## 进入主题前（2 分钟）

Day 2 到 `slot_mapping`
就停了：`position → logical block → physical block → flat slot`。今天闭环
execution side：attention backend 拿到本轮 K/V 后，怎样消费 flat
slot，把 token 写进 paged cache tensor。

只盯四个对象：`key/value`、`slot_mapping`、`block_size`、`key_cache/value_cache`。

``` text
block_idx    = slot // block_size
block_offset = slot %  block_size
```

## 前置数据结构（3 分钟）

`key/value`：当前 forward 新算出的
`[num_tokens, num_kv_heads, head_size]`。

`slot_mapping`：`[num_tokens]`；第 `t` 个值告诉 kernel token `t`
应写到哪个 flat cache slot。

普通 Triton cache 主路径先看作
`[num_blocks, block_size, num_heads, head_size]`。

两个 invariant： 1. `slot < 0` 是 padding/无效 token，不能写 cache。 2.
`slot_mapping` 只决定"哪一 block、哪一 slot"；head/dim 内部 layout 由
cache tensor stride 决定。

## 最小流程

``` text
BlockTable.compute_slot_mapping()
        ↓
slot_mapping
        ↓
attention backend
        ↓
triton_reshape_and_cache_flash(...)
        ↓
reshape_and_cache_kernel_flash
        ↓
slot → (block_idx, block_offset)
        ↓
stride address
        ↓
tl.store K/V
```

## 源码阅读导航（15 分钟，约 120 行）

文件：`vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`

<https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py>

### Step 1：kernel 地址主路径（8 分钟）

搜索：

``` python
def reshape_and_cache_kernel_flash(
```

从参数读到第一次 K/V `tl.store(...)` 完成后停止。只追：

``` text
token_idx
slot_idx
slot_idx < 0
block_idx
block_offset
普通 4D layout 的 tgt_idx_k/tgt_idx_v
key_load/value_load
tl.store
```

跳过：head-major、FP8 scale、TILE_SIZE 调优、per-token-head
quantization、DiffKV、ROCm/XPU。

普通 4D layout 的目标地址本质：

$$
addr = base
+ block\_idx \cdot stride_{block}
+ block\_offset \cdot stride_{page}
+ head \cdot stride_{head}
+ dim
$$

### Step 2：wrapper launch contract（5 分钟）

搜索：

``` python
def triton_reshape_and_cache_flash(
```

只读函数参数 → `block_size/strides` → grid → kernel launch。确认 wrapper
把 tensor strides 与 `slot_mapping` 一起交给 kernel，所以 kernel
不需要知道 `BlockPool`、`Request`、`Scheduler`。

### Step 3：backend 调用点（2 分钟）

文件：`vllm/v1/attention/backends/triton_attn.py`

<https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py>

搜索 `triton_reshape_and_cache_flash(`，只看调用点周围，确认传入
`key/value/key_cache/value_cache/slot_mapping/dtype/scale`，然后停止。

## 自检

1.  为什么 `slot_mapping` 能解耦 Scheduler/BlockPool 与 cache-write
    kernel？
2.  为什么 `slot_mapping` 一维，而 cache tensor 可以是四维/五维？
3.  为什么 `slot < 0` 必须最先处理？
4.  cache 改成 head-major 后，为什么 `slot_mapping` 语义可以不变？

# Part 3 --- 面试（约 11 分钟）

## 口述：为什么 allocator block size 与 kernel layout 可以解耦？

allocator 关心 request ownership、复用、eviction、fragmentation；kernel
关心 tensor layout 和访问效率。中间用
`block_table / slot_mapping / stride metadata`
翻译。只要执行侧最终得到正确
`(physical block, in-block offset)`，allocator 不需要知道 head/dim
排列，kernel 也不需要理解 request/refcount/hash。

追问：为什么不让 kernel 直接查
Request/BlockPool？因为会把高层动态状态泄漏到设备执行层，metadata 与
CPU/GPU 边界复杂，也更难 CUDA Graph 化。

## 手撕：CUDA `float4` copy + tail

``` cpp
#include <cuda_runtime.h>

__global__ void copy_float4(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_n = n / 4;
    if (tid < vec_n) {
        reinterpret_cast<float4*>(dst)[tid] =
            reinterpret_cast<const float4*>(src)[tid];
    }
    int tail = vec_n * 4;
    if (tid == 0) {
        for (int i = tail; i < n; ++i)
            dst[i] = src[i];
    }
}
```

必须主动补：这个版本假设 `src/dst` 满足 `float4` 对齐；任意偏移 pointer
不一定满足。

# 今日验收

1.  能一句话区分 `TMA GMEM→SMEM` 与
    `tcgen05.cp SMEM→TMEM`，并知道后者不是每个 dense GEMM 必经。
2.  能指出 `slot → block_idx/block_offset → stride address → tl.store`
    完整写路径。
3.  能解释为什么 KV tensor layout 改变时 `slot_mapping` 的语义仍可不变。

**下一课：`tcgen05.mma` operand/accumulator 数据流与 single-thread
issue；KV/state 侧进入 paged cache read path。**
