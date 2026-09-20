# AUDIT

## 内容与事实检查

- GH100 完整实现：8 GPC、72 TPC、9 TPC/GPC、2 SM/TPC、144 SM；H100 SXM5：8 GPC、66 TPC、132 SM、10 个 512-bit memory controllers、50 MB L2。均与 NVIDIA Hopper 官方技术博客/白皮书交叉。
- A100 L2：40 MB、位于 GPC 外、两个高层 partition、每个 partition 40 个 512 KB slices、每个 controller 关联 8 slices。与 NVIDIA A100 白皮书交叉。
- Hopper cluster：同一 cluster 的 blocks 同时调度到单一 GPC；portable 最大 cluster size 为 8。与当前 CUDA Programming Guide 交叉。
- 专利 US20230315655A1 的申请/优先权日 2022-03-10、公开日 2023-10-05、NVIDIA assignee 已核对；正文明确标为 patent evidence / implementation not confirmed。

## 架构语义检查

- 解释图没有把 L2/HBM 画进 SM/GPC 的简单父子树，而是分为 compute replication side 与 shared memory backend。
- GPC-local SM-to-SM path 只标给 cluster/DSM，不暗示所有 global-memory traffic 绕过 L2。
- SM→L2→controller/HBM 图把 coalescing 与 address mapping 分开，并明确 hash/router/arbitration 未公开。
- 未把 TPC 说成 CUDA launch 可寻址对象，未把 memory partition、L2 partition、MIG partition 混为同一粒度。

## 渲染可读性检查

- 两张 SVG 均采用 1400 px 宽画布、浅底深字、高对比边框与箭头；原先带 BOM 的文件已重写，避免部分 SVG 解析器显示空白。
- 按元素坐标检查标题、说明、方框、箭头和页脚均在 viewBox 内；用 ImageMagick 预览验证布局，中文字体由渲染器回退处理。
- 官方 raster 图已下载并在 `REMOTE_IMAGES.md` 登记来源、尺寸和 SHA-256。

## Markdown 检查

- 一级标题唯一；标题层级连续。
- 本地解释图使用 `assets/...` 相对路径；官方图使用本地副本并保留来源记录。
- 公式使用 Typora 兼容的 `$...$` / `$$...$$` 定界符；UTF-8。
