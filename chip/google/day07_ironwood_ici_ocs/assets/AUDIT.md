# AUDIT

Round 1 — 事实审核：64-chip 4×4×4 cube、完整 cube/典型 torus 节点的 6-direction adjacency、cube 内 copper ICI、cube 间 optical ICI + OCS、256-chip=4 cubes、9216-chip=144 cubes、DCN beyond superpod 均对照 Google Cloud 一手资料。GKE topology 数量对照当前官方表格；小 slice 的退化边界不机械套用六个不同邻居。

Round 2 — 架构语义审核：明确区分 cube 与 slice；remote DMA reachability 与 direct adjacency；OCS circuit switching 与 packet switching；未把 NVSwitch 与 OCS 声称为等价拓扑。

Round 3 — 渲染审核：本地 SVG 画布、文字边界、箭头方向、层次标签检查完成；标注 schematic/not to scale；未公开 OCS 内部细节标 not disclosed。

官方资料图 `01_ironwood_slice_topologies_official.jpg` 与 `02_ironwood_3d_torus_official.png` 的来源页面、原始 URL、抓取日期、尺寸、SHA-256 和版权边界已登记在 `REMOTE_IMAGES.md`。Google Cloud Blog 未对这两张嵌图单独声明 CC BY/Apache 许可证，因此正文只作带来源的学习引用，不将其标为开源素材。
