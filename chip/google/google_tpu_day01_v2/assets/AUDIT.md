# SVG 审核记录

本学习包的四张 SVG 均为教学重绘，不宣称是 Google 原始图。审核按三轮进行。

## A. 事实审核

- `03_ironwood_package_reconstructed.svg`
  - 对照 TPU7x 官方文档：每 chip 含 2 TensorCores、4 SparseCores、192 GB HBM。
  - 对照 dual-chiplet 文本：每 chiplet = 1 TensorCore + 2 SparseCores + 96 GB HBM；两个 chiplet 独立 memory space。
  - 对照官方架构图：保留 Host、gBMC、PCIe Gen5 x16 / Gen2 x1、TensorCore 内的 TCS/XLU/MXU/VPU+VMEM、Memory & DMA Interconnect、SparseCore、HBM controllers/stacks、SerDes chiplet、ICI/ICR Router、link stack、SerDes。
  - 对照官方文字：D2D 约为单个 1D ICI link 的 6×。
  - 未绘制官方未披露的真实位宽、时钟、floorplan、内部 arbitration。

- `02_ironwood_resource_hierarchy.svg`
  - 1 compute chiplet 对应框架中的 1 device；1 TPU7x chip 包含 2 chiplets，因此框架可见 2 devices/chip。
  - 1 TPU7x VM = 4 chips、224 vCPU、960 GB RAM、2 NUMA nodes。
  - Pod = 9216 chips；示例 slice topology 与官方表一致。

- `04_tpu8_training_vs_inference.svg`
  - TPU 8t：216 GB HBM、6528 GB/s、12.6 PFLOPS FP4、3D torus、9600-chip superpod、SparseCore。
  - TPU 8i：288 GB HBM、8601 GB/s、384 MB SRAM、10.1 PFLOPS FP4、Boardfly、CAE；CAE 替换此前 core dies 上的 4 SparseCores。
  - 图仅表达官方公开的功能分工，不推测 microarchitecture 未公开细节。

## B. 架构语义审核

- 资源层级图明确标注为“逻辑/资源层级”，避免被误读为物理布线。
- package 图中的 compute chiplet、SerDes chiplet 和 host interface 分层，不把 JAX device 与 physical chip 混为一谈。
- TPU 8t/8i 对比图是教学归纳，不把 memory / network 框画成 die 内真实物理位置。

## C. 可读性审核

- 所有 SVG 使用 `viewBox`，Typora/浏览器缩放可保持比例。
- 关键模块字号 >= 13px，标题 >= 20px。
- 图中文字不依赖 hover；没有交叠文本。
- 箭头只用于确定的连接/层级；抽象关系用分组框而非伪造物理连线。
- 后续已渲染为 PNG 进行人工检查；若发现遮挡或越界会回改 SVG。

## 审核后的修改记录

- 第一次渲染后发现 `03_ironwood_package_reconstructed.svg` 中 Chip Manager 没有明确接入管理/内存路径，容易造成“孤立模块”的误读；第二版补上了 PCIe/Chip Manager 到 chiplet A `Memory & DMA Interconnect` 的示意连接。
- 第二次渲染后发现 D2D 标签压在 chiplet 边界与 interconnect 附近；第三版把标签上移，避免把文字误读成总线本身。
- `04_tpu8_training_vs_inference.svg` 初版的 `embedding / irregular gather` 与 `reduction / synchronization` 标签过长；第二版拆成两行，避免越界。
- `01_tpu_generations_timeline.svg` 与 `02_ironwood_resource_hierarchy.svg` 首轮渲染未发现遮挡、层级歧义或越界，保留。
