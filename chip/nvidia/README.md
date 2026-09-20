# NVIDIA 硬件与系统学习文档

这组笔记从整机通信域逐步推进到单 GPU 的计算层级和 memory path，再用 profiler counter 定位带宽瓶颈。每篇都区分产品官方图、教学重绘和未公开的物理实现。

| 篇 | 主线 | 文档 |
|---|---|---|
| Day 13 | GPU、NVLink/NVSwitch、PCIe、NIC 与 scale-out 通信域 | [day13_nvidia_platform_system_map.md](day13_nvidia_platform_system_map/day13_nvidia_platform_system_map.md) |
| Day 14 | GPC、TPC、SM、L2、memory partition 与 Hopper cluster 边界 | [day14_nvidia_gpc_tpc_sm_l2_memory_partition.md](day14_nvidia_gpc_tpc_sm_l2_memory_partition/day14_nvidia_gpc_tpc_sm_l2_memory_partition.md) |
| Day 15 | coalescing、sector、L2 fabric、controller 到 HBM 的诊断路径 | [day15_nvidia_memory_fabric_path.md](day15_nvidia_memory_fabric_path/day15_nvidia_memory_fabric_path.md) |

## 读图约定

- 官方产品图保留在对应 `assets/`，来源、直链、抓取日期和 SHA-256 写在 `REMOTE_IMAGES.md`。
- `schematic / not to scale` 的 SVG 只表达职责、数据依赖或诊断顺序，不代表 NVIDIA 未公开的 floorplan、地址 hash、router hop 或仲裁实现。
- `GB/s`、`Gbps`、sector、cache line 和 `% peak` 的作用域必须跟随正文解释，不能只看图中数字。

推荐顺序是 Day 13 → Day 14 → Day 15：先确定数据跨越哪个通信域，再进入 GPU 内部层级，最后用 Nsight Compute 的可观测指标定位最早异常。
