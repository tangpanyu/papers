# Day 06 资料与图示审核记录

## 事实审核

- 源码基线固定为 SGLang fork commit [`db017e34902b51e1fd1ac7ebbedaf720c75b374d`](https://github.com/tangpanyu/sglang/commit/db017e34902b51e1fd1ac7ebbedaf720c75b374d)；正文的行号链接均指向该 commit。
- `ReqToTokenPool`、`RadixCache`、MLA pool、`IndexKeyCache`、DSA metadata 和 top-k transform 的形状/所有权语义按该快照核对。`page_table_1` 明确限定为 eager/legacy 路径；fused top-k v2 的 compact real-page-table 路径单独标注。
- 本文不把该 fork 中不存在的 GLM-5.3 model-specific glue 写成端到端实现。

## 语义审核

- 红色实线表示 persistent tensor payload，蓝色虚线表示 per-request/per-forward metadata，绿色表示 ownership/lifecycle；三种关系不互相冒充数据复制。
- latent 主池与 indexer sidecar 共用 flat location 和 allocator 生命周期，但不共享字节；sidecar page 的物理布局是连续 K 区 `[page_size, index_head_dim]` 后接连续 scale 区（SoA），同一 token 通过 page offset 配对，不是交错的 132-byte AoS 记录；top-k 只属于当前 forward。
- `topk_transform` 的真正 unfused 分支返回窗口局部索引（selected columns 减 `row_starts`；无窗口起点时才等于 raw columns）；只有 legacy fused PAGED 路径读取 size-1 `page_table_1`，满足 v2 条件的 fused decode 直接读取 compact `real_page_table`。
- 本篇是固定源码版本的软件案例，未强行套用 TPU 硬件图；现有自绘图比通用外部架构图更能表达具体调用链和状态所有权。
- 同时保留 DeepSeek-V3.2-Exp 论文 Figure 1 的官方 DSA-under-MLA 架构图作为模型级语义锚点；该图只说明 indexer/top-k/core-attention，不替代本篇针对 SGLang allocator/Radix/pool 的自绘状态图。

## 渲染审核

- 三张 SVG 已实际渲染检查 `viewBox`、文字边界、箭头方向和浅/深背景可读性；均标注为教学示意，不代表真实 GPU/TPU floorplan。
- 论文图的 PNG 裁剪副本已检查清晰度与边界，保留来源/裁剪说明，不把裁剪图误称为本仓库原创。
