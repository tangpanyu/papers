# 本地资料图记录

本课新增两张来自 JAX 官方文档/仓库的资料图；原文仍保留本目录的自绘 `01_double_buffer.svg`，用于把抽象 API 对应到 A/B slot。

| 文件 | 用途 | 来源页面 | 原始文件 | 抓取日期 | 尺寸 / 校验 |
|---|---|---|---|---|---|
| `03_tpu_memory_space_jax_official.png` | HBM、VMEM、SMEM、VREG/SREG 与 MXU/VPU/Scalar 的关系 | [JAX Pallas — TPU Pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html) | 页面内嵌 `TPU Memory Space Cartoon.png`（data URI） | 2026-09-08 | 474×446 RGBA PNG; SHA-256 `9690c79c506b4175e64b83bb6477f43a45e52c34076b291e6cefb5398f7c7a66` |
| `02_pipelining_bandwidth_bound_official.svg` | copy/compute 重叠与 bandwidth-bound idle | [JAX — Software Pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html) | [JAX repository asset @ `02fed78`](https://raw.githubusercontent.com/jax-ml/jax/02fed78da8636337dcc051079c03947cd9949908/docs/_static/pallas/pipelining_bandwidth_bound.svg) | 2026-09-08 | viewBox 1100×140; SHA-256 `05eb51cc23ba83cec5793838f11a01f10a468bcc21f4e4d6c8c1ad94cc01295d` |

## 使用与许可说明

- 两张图均未裁剪或改色；正文通过本地相对路径引用，避免远程 URL 失效。
- JAX 仓库以 [Apache License 2.0](https://github.com/jax-ml/jax/blob/02fed78da8636337dcc051079c03947cd9949908/LICENSE) 发布；图一是文档页面内嵌的 JAX Authors 图，页面标注 `© Copyright 2024, The JAX Authors`。再分发时保留作者与来源链接，不把图标作本仓库原创。
- `02_pipelining_bandwidth_bound_official.svg` 的语义是通用 software-pipeline 时间线，不能据此推断某一 TPU 型号的 cycle、带宽或物理布局。
