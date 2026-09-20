# 本地资料图记录

`01_pallas_lowering_path_official.png` 是 JAX Pallas Design 文档中的官方 lowering-path 图；`02_mosaic_pipeline_schematic.svg` 是本课根据公开资料绘制、用于补充 LLO/backend 责任边界的教学图。

| 文件 | 来源页面 | 原始文件 | 抓取日期 | 尺寸 / 校验 |
|---|---|---|---|---|
| `01_pallas_lowering_path_official.png` | [Pallas Design](https://docs.jax.dev/en/latest/pallas/design/design.html) | [JAX repository asset @ `02fed78`](https://raw.githubusercontent.com/jax-ml/jax/02fed78da8636337dcc051079c03947cd9949908/docs/_static/pallas/pallas_flow.png) | 2026-09-08 | 908×832 RGBA PNG; SHA-256 `9daf85d6236b5c87b019795c71f510c186d885a7a1827ddffa6d896f472cd4d5` |

## 使用与许可说明

- 官方图以原始透明背景和标注保存；正文说明它是 lowering path 总览，不把图中的代码框当作最终 TPU assembly。当前 JAX Design 文档已将图中的 Pallas→Triton GPU 分支标为历史/弃用路径；本课只采用 TPU→Mosaic 分支的语义。
- JAX 仓库提供 [Apache License 2.0](https://github.com/jax-ml/jax/blob/02fed78da8636337dcc051079c03947cd9949908/LICENSE)。再分发时保留 JAX Authors、来源页面和许可证链接。
