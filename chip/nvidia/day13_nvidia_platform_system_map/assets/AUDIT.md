# Day 13 图像与事实审计

## 01_nvidia_communication_domains.svg

- 类型：自绘解释图，标注 `schematic / not to scale` 与 `not disclosed`。
- 事实检查：NVLink/NVSwitch 为 scale-up；PCIe 承担 host/I/O 与 GPU-NIC peer-memory path；InfiniBand/Ethernet 为 scale-out；Grace 产品可用 NVLink-C2C 取代传统 CPU-GPU PCIe 数据路径。
- 语义检查：三行表示通信 domain，不表示实际 floorplan、带宽比例或所有 SKU 的固定 wiring。
- 可读性检查：1700×980；按 viewBox/元素坐标检查并用本地 SVG 预览确认，文字未截断。

## official_dgx_b200_system_topology.png

- 类型：NVIDIA DGX B200 User Guide 官方拓扑图。
- 原始 URL：`https://docs.nvidia.com/dgx/dgxb200-user-guide/_images/dgx-b200-system-topology.png`。
- 文件检查：805×461，92709 bytes，PNG RGBA。
- 语义边界：只用于读取官方明示的 CPU、GPU、NVSwitch、PCIe switch、ConnectX-7、BlueField-3、NVMe 与连接关系；不从线条颜色反推未标出的协议细节。

## official_gb200_nvl72.jpg

- 类型：NVIDIA GB200 NVL72 官方产品图。
- 原始 URL：`https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/gb200-superchip/gb200-nvl72-og.jpg`。
- 文件检查：1200×630，48152 bytes，JPEG。
- 用途：展示 rack-scale liquid-cooled product boundary；结构与带宽事实来自同页文字规格，而非从照片推断。

## 02_collective_path_hierarchy.svg

- 类型：自绘解释图，标注 `schematic / not to scale`。
- 事实检查：collective 同时涉及 local HBM traffic、NVLink/NVSwitch scale-up 和 NIC/IB/Ethernet scale-out；NVLS 标为 runtime 可选路径，不声称所有系统或消息使用。
- 语义检查：三个框是物理层次，不是固定串行 timeline；正文明确 NCCL 可 chunk、pipeline 和 overlap。
- 可读性检查：1700×900；按 viewBox/元素坐标检查并用本地 SVG 预览确认，文字未截断。

## 三轮结论

- 事实：通过；产品专属事实均标明 DGX B200 或 GB200 NVL72，没有外推为所有 NVIDIA SKU。
- 架构语义：通过；PCIe、NVLink/NVSwitch、NVLink-C2C 和 scale-out network 已分层。
- 渲染可读性：通过；两张 SVG 和两张官方 raster image 均可离线打开。
