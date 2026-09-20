# Day 15 图像与事实审计

## 图像来源

- `official_nsight_l2_model.png`：NVIDIA Nsight Compute Profiling Guide 的 GA100 L2 cache model，原始 URL 为 `https://docs.nvidia.com/nsight-compute/_images/hw-model-lts-ga100.png`，2026-09-18 下载。
- `official_coalesced_access.png`：NVIDIA CUDA C++ Best Practices Guide 的 coalesced access 图，原始 URL 为 `https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/coalesced-access.png`，2026-09-18 下载。
- `01_request_to_hbm.svg`、`02_counter_diagnosis.svg`：2026-09-18 重绘的浅色解释示意，根据 NVIDIA 官方 CUDA 与 Nsight Compute 文档，不表示量产芯片 floorplan。

## 三轮检查

1. 事实检查：区分 warp request、32 B sector、128 B L2 line、L2 miss 与 device memory traffic；未公开的地址 hash、物理拓扑与 controller 映射均标为 `not disclosed` 或推断。
2. 架构语义检查：L1/shared 是 SM-local；L2 与 memory controller 属于共享后端；coalescing、cache locality、fabric/partition balance、HBM saturation 是不同问题。
3. 渲染检查：两张 SVG 使用固定 1400 px 画布、浅底深字与足够对比度；用 ImageMagick 预览检查标题、路径和箭头均在画布内。

## 证据边界

- Nsight Compute 官方文档可证明 profiler 暴露 L1-to-XBAR、L2、L2 fabric 与 device-memory 等路径的流量/利用率，但不等于 NVIDIA 公开了实际 NoC topology。
- 2025 Hopper microbenchmark 论文报告的 near/far L2 latency 是作者在 A100/H800 上的实验解释，不能直接外推到所有 H100/H200/B200 SKU。
- 专利 `US20230289189A1` 仅作为 L2/processing resources 可随 GPU partition 切分的 embodiment 证据，不能证明 Hopper 的地址 hash 或 floorplan。
