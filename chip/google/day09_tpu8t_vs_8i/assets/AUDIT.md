# AUDIT

Round 1 — 事实审核：TPU 8t/8i 当前状态核对 Google Cloud TPU 产品页，2026-09-06 仍标记 Coming soon；规格和 Boardfly/CAE 数据核对 2026-04-23 Google Cloud 架构 deep dive。

Round 2 — 架构语义审核：没有把 8t/8i 简化成“训练算力版/推理低算力版”；把 SRAM、CAE、Boardfly、HBM、SparseCore 和 topology 作为系统级 specialization 处理。Boardfly 的 16→7 hops 只用于 Google 给出的 1024-chip 对照，不泛化到任意规模。

Round 3 — 渲染审核：三张本地 SVG 与四张官方 PNG 检查画布、文字边界、连线、标签和层级；解释图均标注 schematic / not to scale，官方图已下载并在 REMOTE_IMAGES.md 登记来源与校验。
