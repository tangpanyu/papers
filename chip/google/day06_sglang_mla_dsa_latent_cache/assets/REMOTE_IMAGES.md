# 本地资料图记录

本课新增一张模型作者发布的 DSA-under-MLA 架构图；SGLang 的 allocator、Radix ownership 和 page 生命周期仍由本目录的自绘 SVG 表达。

| 文件 | 用途 | 来源页面 | 原始文件 | 抓取日期 | 尺寸 / 校验 |
|---|---|---|---|---|---|
| `01_deepseek_dsa_mla_official.png` | Lightning Indexer、Top-k Selector 与 MLA Core Attention 的模型级数据流 | [DeepSeek-V3.2-Exp 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | [DeepSeek_V3_2.pdf @ commit `87e509a`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/DeepSeek_V3_2.pdf)，第 2 页 Figure 1 | 2026-09-09 | 1700×1000 PNG; SHA-256 `1886ee6bc9dd1409448e3d018af75760715d9d18827c337a4b9e40866289e75d` |

## 提取与许可说明

- PNG 是从论文 PDF 第 2 页按 250 dpi 渲染后裁剪的学习用副本（裁剪框：渲染像素 `x=180,y=250,w=1700,h=1000`；隐藏 PDF 链接注释），不是对图内容的重新绘制。论文原始 Figure 1 的完整 caption 保留在图底部。
- 官方仓库包含 [MIT License](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/LICENSE)。再分发时同时保留 DeepSeek-AI、论文标题、仓库/commit 链接；若下游项目对论文图另有版权政策，以原作者要求为准。
- 该图表达模型级 attention 数据流，不包含本课固定 SGLang snapshot 的具体类名、page size 或 allocator ownership；正文明确区分两种证据。
