# Google TPU 学习文档

这组笔记把 Google TPU 从单芯片执行单元一路串到切片互连，并在最后加入一个 SGLang 运行时案例。建议按下面的顺序阅读：

| 篇 | 主题 | 文档 |
| --- | --- | --- |
| Day 01 | 从 chiplet、chip 到 VM / slice / Pod 的系统地图 | [day01_google_tpu_system_map.md](google_tpu_day01_v2/day01_google_tpu_system_map.md) |
| Day 02 | TensorCore、MXU 与 JAX/Pallas 执行模型 | [day02_tpu_tensorcore_execution_model.md](google_tpu_day02_regenerated/day02_tpu_tensorcore_execution_model.md) |
| Day 03 | HBM、VMEM/SMEM 与双缓冲流水 | [day03_tpu_memory_pipeline.md](day03_tpu_memory_pipeline/day03_tpu_memory_pipeline.md) |
| Day 04 | BlockSpec、layout 与 MXU tiling | [day04_block_layout_mxu_tiling.md](day04_block_layout_mxu_tiling/day04_block_layout_mxu_tiling.md) |
| Day 05 | Pallas 到 Mosaic/低层编译流水 | [day05_mosaic_compiler_pipeline.md](day05_mosaic_compiler_pipeline_fresh/day05_mosaic_compiler_pipeline.md) |
| Day 06 | SGLang MLA/DSA latent KV cache（软件实现案例） | [day06_sglang_mla_dsa_latent_cache.md](day06_sglang_mla_dsa_latent_cache/day06_sglang_mla_dsa_latent_cache.md) |
| Day 07 | Ironwood 的 ICI、3D torus、OCS 与 slice | [day07_ironwood_ici_ocs.md](day07_ironwood_ici_ocs/day07_ironwood_ici_ocs.md) |
| Day 08 | TPU VM、host attachment、ICI/DCN 边界 | [day08_tpu_host_ici_dcn_boundary.md](day08_tpu_host_ici_dcn_boundary/day08_tpu_host_ici_dcn_boundary.md) |
| Day 09 | TPU 8t / 8i 的训练与推理专用化、CAE、Boardfly | [day09_tpu8t_vs_8i.md](day09_tpu8t_vs_8i/day09_tpu8t_vs_8i.md) |
| Day 10 | SparseCore、CAE 与不规则工作旁路 | [day10_tpu_specialized_bypass_engines.md](day10_tpu_specialized_bypass_engines/day10_tpu_specialized_bypass_engines.md) |
| Day 11 | JAX、Shardy、XLA、Pallas、PJRT、Pathways 执行链 | [day11_google_tpu_software_execution_chain.md](day11_google_tpu_software_execution_chain/day11_google_tpu_software_execution_chain.md) |
| Day 12 | 从算子到 TPU 集群的系统复盘 | [day12_google_tpu_system_recap.md](day12_google_tpu_system_recap/day12_google_tpu_system_recap.md) |

## 读图和读数约定

- 文档中的动态规格、产品状态和 API 以各篇开头标注的“资料复核”日期为准；Day 06 的源码事实例外，以文中固定 commit 为准。产品状态区分 GA、预览和 Coming soon。
- `GB`/`GB/s` 使用十进制，`GiB` 使用二进制；`Gbps` 是 bit/s，不能直接当成 `GB/s`。表格会注明是 per TensorCore、per chip、per VM 还是 per slice。
- 官方图片保留在对应目录的 `assets/` 下，并在 `REMOTE_IMAGES.md` 记录来源、直链、检索日期和许可性质。许可不明的图片仅作学习引用，不把它们描述为可自由再分发素材。
- 标有“教学重绘 / 示意 / not to scale”的 SVG 只表达阅读路径，不代表公开的 floorplan、lane 或协议细节。

Day 01–05 主要建立硬件与编译器心智模型；Day 06 是固定源码版本上的软件案例；Day 07–10 追踪互连、主机边界和训练/推理专用化；Day 11–12 把硬件重新接回软件执行链并收束全局。各篇末尾的“下一跳”按这个目录组织，而不是把逻辑层级当成物理连线。
