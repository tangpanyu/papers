# Day 12 图像与事实审计

## 01_tpu_unified_system_map.svg

- 类型：自绘解释图，标注 `schematic / not to scale / 根据 Google 官方资料重绘`。
- 事实检查：ICI 属 slice 内通信；multislice 超出 ICI connectivity 后使用 DCN；OCS 用于 cube 间光学 ICI 重构；Pallas/Mosaic 与 Pathways 分别位于局部 kernel 与高层 orchestration 边界。
- 语义检查：箭头表达职责和数据路径，不表达物理 floorplan、带宽比例或实际 NoC。
- 可读性检查：1600×900，文字字号不低于 16 px；浅色背景与深色文字对比通过。

## official_tpu_v4_pod.png

- 类型：Google Cloud 官方照片，原始 URL：`https://storage.googleapis.com/gweb-cloudblog-publish/images/1_Cloud_TPU_v4.max-1100x1100.jpg`。
- 实际格式：服务器返回 PNG，已按真实格式保存为 `.png`；1100×313，764160 bytes。
- 用途：说明逻辑 torus/cube 最终落到机架、布线、供电与散热的物理系统，不从照片反推未公开拓扑。

## 02_workload_path_training_decode.svg

- 类型：自绘解释图，标注 `schematic / not to scale`。
- 事实检查：训练/推理路径为典型压力模型；TPU 8t 的训练取向、TPU 8i 的大 SRAM/CAE/Boardfly 来自 Google 2026-04-22 官方技术介绍。
- 语义检查：箭头表示主要压力向系统边界传递，不声称所有 workload 顺序执行这些阶段。
- 可读性检查：1600×850，训练与 decode 分成两行，颜色和文字均可独立区分。

## official_tpu8t_asic.png

- 类型：Google Cloud 官方 TPU 8t ASIC block diagram。
- 原始 URL：`https://storage.googleapis.com/gweb-cloudblog-publish/images/1_v4.max-1600x1600.png`。
- 文件检查：1592×1132，135029 bytes，PNG RGBA。
- 边界：只引用图中明示的 TensorCore、SparseCore、Memory and DMA Interconnect、HBM controller、ICI router/link stack、SerDes chiplet 与 host interface；不从框图位置推断 floorplan。

## 03_google_nvidia_translation_map.svg

- 类型：自绘跨平台解释图，标注职责近似而非结构等价。
- 事实检查：Google 侧依据本课参考资料；NVIDIA 侧只是后续课程的证据路线与主题预告，不包含未经验证的具体微架构事实。
- 语义检查：虚线标“共同问题”，明确禁止一一对应。
- 可读性检查：1600×880，左右对照，中心标签显示比较维度。

## 总体结论

- 事实：三轮检查通过；路线图产品状态在正文中单独限定。
- 架构语义：三轮检查通过；所有自绘图均未表达未公开 floorplan。
- 渲染可读性：三轮检查通过；SVG 使用内嵌字体回退和固定 viewBox。

