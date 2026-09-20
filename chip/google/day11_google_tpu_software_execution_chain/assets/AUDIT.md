# AUDIT

Round 1 — 事实审核：
- JAX AI stack 当前包含 JAX/XLA/Pathways/Pallas/MaxText/Tunix/vLLM/XProf：核对 2026-08 Google Cloud JAX AI stack 文档。
- 当前 JAX `Sharding` 表示数组在物理设备内存上的分布，`Mesh + PartitionSpec` 为核心表达：核对当前 JAX sharding 文档。
- JAX Shardy migration 文档明确在 2026-03 完成后 Shardy 成为 JAX 唯一 partitioner，取代 GSPMD/PartIR 的用户侧 partitioning 路线。
- PJRT 为 framework/hardware-independent device API，PjRtClient 管设备/内存空间、loaded executable 负责 Execute：核对 OpenXLA 2026-06 文档。
- Pathways 是大规模 accelerator orchestration layer；论文使用 asynchronous sharded dataflow、gang scheduling，并在 2048 TPU SPMD 上达到接近 100% utilization：核对 MLSys 2022 Google Research。
- 2026 Pathways-on-Cloud 文档说明 JAX with Pathways 通过 proxy backend 把所有集群 TPU 视为 local，`jax.process_index()` 为 0。
- TPU 8 官方 deep dive 明确写 XLA 隐藏 Boardfly topology / CAE synchronization，并支持 Pallas/Mosaic 触达 CAE/SparseCore。

Round 2 — 架构语义审核：
- 明确区分普通 multi-controller JAX 与 Pathways；不把 Pathways画成所有 JAX 程序必经 runtime。
- 不把 `Mesh/PartitionSpec` 说成物理路由；它是 logical placement/sharding intent。
- 不把 XLA 自动 collectives 与 Pallas explicit remote DMA 混为一层。
- 不声称用户手写 op 名就直接选择 CAE/SparseCore；映射依赖 compiler/library/kernel support。

Round 3 — 渲染审核：
- 三张本地 SVG 检查画布、文字边界、层次、箭头。
- 全部标注 schematic / not to scale，并明确 Pathways optional。
