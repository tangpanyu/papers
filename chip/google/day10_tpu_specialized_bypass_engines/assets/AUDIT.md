# AUDIT

Round 1 — 事实审核：TPU v4 SparseCore 5–7× embedding-heavy model speedup、约 5% die area/power核对 ISCA 2023/arXiv 2304.01433；TPU 8t SparseCore 与 TPU 8i CAE 的职责、CAE replacing four SparseCores、5× on-chip collective latency核对 Google Cloud 2026 TPU 8 technical deep dive；JAX TPU Embedding 的软件可见职责核对 jax-ml/jax-tpu-embedding。

Round 2 — 架构语义审核：不把 SparseCore 写成稀疏矩阵乘核心；不把 CAE 写成 Boardfly/NVSwitch 类交换结构；不推断未公开 tile/router/register/PHY 细节。

Round 3 — 渲染审核：三张 SVG 检查画布、文字、箭头、标题；均标明 schematic/not to scale 或未公开边界。
