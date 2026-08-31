# Day 1 / Day 2 重构说明

这版重点不是增加内容，而是严格控制阅读边界。

## 源码阅读节点统一规格

`ai_infra/` 下各学习正文的 PTX、CUTLASS/CuTe 与 vLLM 源码阅读节点统一按下面顺序组织：

1. **前置条件**：直接放在对应 Step / 精确阅读条目下面，只写进入该段源码前必须已经成立的输入、状态和简化条件。
2. **本步目的**：紧跟前置条件，明确该段只完成什么、产出什么，以及哪些问题不在本步解决。
3. **原阅读范围**：随后保留直达链接、搜索词、代码范围、跳过项和自检，不把补充说明集中追加到文档末尾。
4. **图示**：保留 Markdown/Mermaid；本地图片统一放在相邻 `assets/` 目录并使用相对路径。

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

所有嵌入图片都已保存到 [`assets/`](assets/)，Markdown 使用相对路径，不依赖联网加载。图片文件与官方来源的对应关系见 [`assets/README.md`](assets/README.md)。
