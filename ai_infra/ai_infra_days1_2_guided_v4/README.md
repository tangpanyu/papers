# Day 1 / Day 2 重构说明

这版重点不是增加内容，而是严格控制阅读边界。

## 源码版本

vLLM pinned commit:

`80771bbbddf9e5153eea3aca8055049ee5aaaed1`

日期：2026-08-25。

## 两天源码预算

Day 1:
- `KVCacheBlock` 字段：约 20 行
- `FreeKVCacheBlockQueue` docstring：约 16 行
- `get_new_blocks/_maybe_evict_cached_block`：约 50 行
- `touch/free_blocks`：约 38 行
- 总计约 120 行有效阅读

Day 2:
- `BlockTable` 初始化核心：约 52 行
- `append_row/add_row`：约 20 行
- `compute_slot_mapping` wrapper：约 28 行
- Triton 主计算：约 32 行
- 总计约 130 行

## 图片

Markdown 直接引用以下官方原图 URL，以保证使用原图而非自绘替代：

- NVIDIA PTX Figure 182:
  https://docs.nvidia.com/cuda/parallel-thread-execution/_images/tensor-memory-layout.png
- NVIDIA PTX Figure 183:
  https://docs.nvidia.com/cuda/parallel-thread-execution/_images/tcgen05-mma-fragment-3232b.png
- vLLM Figure 7:
  https://vllm.ai/blog-assets/figures/2025-vllm-anatomy/prefix_pt2.png
- vLLM Figure 8:
  https://vllm.ai/blog-assets/figures/2025-vllm-anatomy/prefix_pt3.png
- vLLM Figure 4:
  https://vllm.ai/blog-assets/figures/2025-vllm-anatomy/fwd_pass.png

当前执行容器无外网 DNS，因此无法把原图 bytes 物理落到 `assets/`；Typora 联网打开 Markdown 时会直接拉取官方图片。
