# AMD CDNA 4 Architecture 白皮书精读

> 原文：[amd-cdna-4-architecture-whitepaper.pdf](../pdfs/amd-cdna-4-architecture-whitepaper.pdf)  
> 文档性质：厂商架构白皮书，不是经过同行评审的论文。文中的性能大多是峰值理论值，不能等同于端到端模型性能。

## 论文一句话总结

CDNA 4 的核心不是简单增加 CU 数量，而是用 **8 个 N3P XCD + 2 个 N6 IOD** 重新平衡整颗 GPU：减少 CU 数量但显著增强每个 CU 的低精度矩阵吞吐、LDS 容量和带宽，同时用 288 GB HBM3E、8 TB/s 带宽及更简单的双 IOD 拓扑喂饱计算单元。

从 AI kernel 视角看，它把主要投资放在了 FP16/BF16/FP8/INT8 和 MXFP4/MXFP6、GEMM 数据复用、低精度转换以及多 GPU 通信上；代价是 FP64 Matrix 峰值减半，且白皮书没有给出真实模型 benchmark、指令 shape 或可持续利用率。

## 1. 背景和问题

大模型训练和推理同时受到计算、HBM 容量、HBM 带宽及 GPU 间通信限制。单纯堆计算单元会导致数据供给不足，单片大 die 又面临良率、成本和工艺不匹配：逻辑适合先进制程，SRAM 与 I/O 却未必从先进制程获益。CDNA 4 因此延续 chiplet 路线，让计算 XCD 使用 N3P，而缓存、HBM 控制器和互联所在的 IOD 继续使用 N6，并围绕低精度 AI 重新分配晶体管预算。

## 2. 核心结论

- 一颗 MI350 系列 GPU 有 8 个 XCD，每个 XCD 物理上有 36 个 CU、启用 32 个，整卡共 256 个 CU；每个 CU 有 4 个 Matrix Core，因此整卡共 1,024 个 Matrix Core。
- MI355X 在 2.4 GHz 下，FP16/BF16 Matrix 峰值为 2.5 PFLOP/s，FP8 和 MXFP8 为 5.0 PFLOP/s，MXFP6/MXFP4 为 10 PFLOP/s；结构化稀疏可让部分格式的理论峰值再翻倍。
- 每 CU 的 LDS 从 64 KB 增至 160 KB，读带宽翻倍至 256 B/cycle，并支持从 L1 直接加载到 LDS。这是让更宽的矩阵流水线真正获得数据复用的关键配套变化。
- 整卡提供 288 GB HBM3E 和 8 TB/s 峰值带宽；256 MB Infinity Cache、每 XCD 4 MB L2、每 CU 32 KB L1 构成分层数据路径。
- 4 个 IOD 合并为 2 个 IOD，并增加 IOD 间直接连接；AMD 声称该路径比上一代快约 14%。NPS2 可把计算和内存限制在同一 IOD 域内，避免跨 IOD 流量。
- 节点内 8 GPU 仍采用全连接拓扑，单 GPU 有 7 条 GPU P2P Infinity Fabric 链路和 1 条到主机的 PCIe Gen5 链路；P2P 峰值聚合带宽为 1,075.2 GB/s。
- 主要代价是架构明显向 AI 倾斜：Matrix FP64 从 CDNA 3 的 256 FLOP/clock/CU 降为 128，即整卡峰值约减半；传统 Vector FP32/FP64 没有每 CU 吞吐提升。
- 最大证据缺口是没有端到端训练/推理结果、功耗归一化结果、MFMA 指令明细、occupancy 数据或真实通信 benchmark。因此只能证明“硬件上限提高了”，不能证明实际工作负载接近该上限。

## 3. 形象解释 / Mental Model

### 3.1 整卡像一座“八车间、双物流中心”的工厂

可以把 8 个 XCD 想成 8 个计算车间，每个车间有 32 个实际投入生产的 CU；两个 IOD 是物流中心，管理 256 MB 公共仓库（Infinity Cache）、8 栈 HBM3E 和对外运输（Infinity Fabric / PCIe）。CDNA 4 没有增加车间数量，甚至比上一代少一些 CU，而是给每条低精度生产线加倍设备，并扩大车间内的周转区 LDS。

baseline CDNA 3 的问题类似“机器升级了，但周转区太小”：矩阵核吞吐提高后，如果每次都从 HBM 搬 A、B tile，计算单元很快会饿死。CDNA 4 把 LDS 从 64 KB 扩到 160 KB，读带宽增至 256 B/cycle，让一个 workgroup 能缓存更大的 A/B tile、做更多次复用，再把结果写回。

真实硬件映射如下：

```text
8 × HBM3E stack（总计 288 GB，8 TB/s）
                ↕
2 × IOD：memory controller + 256 MB Infinity Cache + I/O
                ↕  on-package Infinity Fabric
8 × XCD：每个 4 MB L2 + 32 active CU
                ↕
每 CU：32 KB L1 + 160 KB LDS + vector/scalar/matrix pipelines
                ↕
每 CU 4 × Matrix Core
```

### 3.2 一个 GEMM tile 怎么走

考虑矩阵乘法 $D = AB + C$。一个 workgroup 负责输出矩阵 $D$ 的一个 tile：

1. A、B 最初位于 HBM3E；内存控制器先查询或填充 Infinity Cache。
2. 数据经片上 Infinity Fabric 到目标 XCD，并经过该 XCD 的 4 MB L2。
3. CU 将 A/B tile 经过 L1 搬入显式寻址的 LDS；CDNA 4 可从 L1 直接装载到 LDS，减少用 VGPR 中转。
4. wavefront 从 LDS 取子 tile，送入 Matrix Core 执行 MFMA 类矩阵乘加，累加器通常保留在寄存器中。
5. K 维循环期间反复复用 LDS 中的 A/B 数据；完成后将 D tile 经 L1/L2/Infinity Fabric 写回 HBM。

真正移动的瓶颈不是单一部件，而是“计算和供数的平衡”：低精度 Matrix Core 使计算上限约翻倍，LDS 容量/带宽和 HBM 带宽则避免供数能力停留在上一代。

### 3.3 一个 decode token 怎么走

以单层 Transformer decode 为例：

- 权重 GEMM/矩阵向量乘读取低精度权重；batch 较小时通常更受 HBM 带宽限制，5–10 PFLOP/s 峰值并不会自动兑现。
- attention 读取随上下文长度线性增长的 KV cache。288 GB 容量可放更长上下文或更大 batch，8 TB/s 决定扫描 KV 的上限。
- softmax、归一化和激活依赖向量/超越函数流水线。CDNA 4 将 transcendental rate 提高 2 倍，意图避免 Matrix Core 加速后 softmax 成为明显短板。
- 多 GPU tensor parallel 还需要 all-reduce/all-gather；节点内 7 条 P2P 链路决定通信上限。

如果只记住一张图，应记住原文 Figure 5：**HBM/IOD/Infinity Cache → Infinity Fabric → 8 个 XCD/L2 → 256 CU**。它同时解释了容量、带宽、NUMA、chiplet 和 kernel 数据移动。

## 4. 方法总览

### 4.1 封装与 chiplet

| 层级 | CDNA 4 / MI350 系列 | 含义 |
|---|---:|---|
| XCD | 8 个，TSMC N3P | 放置延迟敏感的计算与低层缓存 |
| IOD | 2 个，TSMC N6 | 放置 Infinity Cache、HBM 控制器和外部互联 |
| CU | 36 个/XCD，启用 32 个/XCD | 4 个冗余 CU 用于良率和频率筛选 |
| 整卡 active CU | 256 | 比 MI300X 少，但每 CU 的 AI 吞吐更高 |
| Matrix Core | 4 个/CU，共 1,024 个 | 执行矩阵乘加 |
| 晶体管 | 185 billion | MI350X 与 MI355X 相同 |

这种分工的本质是：只把逻辑密集、能从缩放获益的 XCD 升到 N3P；SRAM 和高速 I/O 占主导的 IOD 留在成熟的 N6，降低成本和制造风险。

### 4.2 CU 与执行吞吐

每个 CU 仍包含 scalar、vector、matrix 执行管线以及独立的内存管线。代际变化不是增加 CU，而是：

- 16 bit 及以下 Matrix 执行资源翻倍；
- 新增 MXFP8、MXFP6、MXFP4 硬件支持；
- 超越函数速率翻倍，服务 softmax 等算子；
- 增加低精度格式转换指令；
- 扩大 LDS 并提高其带宽。

按白皮书 Table 1，单 CU 每周期理论操作数为：

| 运算 | CDNA 3 | CDNA 4 | 单 CU 变化 |
|---|---:|---:|---:|
| Vector FP64 | 128 | 128 | 1× |
| Vector FP32 | 256 | 256 | 1× |
| Matrix FP64 | 256 | 128 | 0.5× |
| Matrix FP32 | 256 | 256 | 1× |
| Matrix FP16/BF16 | 2,048 | 4,096 | 2× |
| Matrix FP8/INT8 | 4,096 | 8,192 | 2× |
| Matrix MXFP6/MXFP4 | 不支持 | 16,384 | 新增 |

注意：PDF 表格把 MXFP6/MXFP4 写为 `16834`，结合整卡 10 PFLOP/s、256 CU 与 2.4 GHz 反推，应为 **16,384**，这是原文排版错误。

TF32 不再由专用硬件原生支持，而是通过 BF16 软件模拟。它简化了硬件并把资源转向标准低精度格式，但现有依赖 TF32 数值语义或性能的代码需要重新验证。

### 4.3 数值格式：从“一张表一个 scale”到“每 32 个值一个 scale”

普通 FP8 往往为较大 tensor 使用一个 scale。若 tensor 中既有离群值又有小值，共享 scale 会在“溢出”和“把小值量化成零”之间折中。MX 格式通常让连续 32 个低精度值共享一个 8-bit exponent scale：

$$
x_i \approx s_g \cdot q_i, \qquad g = \left\lfloor \frac{i}{32} \right\rfloor
$$

其中 $q_i$ 是 E5M2/E4M3/E3M2/E2M3/E2M1 等低精度值，$s_g$ 是第 $g$ 个 block 的共享尺度。粒度从 whole-tensor 缩到 32-value block，能更贴合局部动态范围，因此 4/6 bit 仍可能保持可用精度。

CDNA 4 支持的重点格式：

- OCP FP8：E5M2 与 E4M3；
- MXFP8：block scale + E5M2/E4M3；
- MXFP6：block scale + E3M2/E2M3；
- MXFP4：block scale + E2M1；
- 传统 FP64、FP32、BF16、FP16、INT8。

这里的关键工程问题不只是 Matrix Core 是否能乘 4-bit 数值，还包括 scale 的加载、block 对齐、量化/反量化和累加精度。白皮书没有给出这些指令级细节。

### 4.4 内存层次

| 层级 | 容量/组织 | 关键特征 |
|---|---:|---|
| CU L1 vector cache | 32 KB/CU，128 B line，64-way | 与上一代基本相同 |
| LDS | 160 KB/CU | 显式寻址；读带宽 256 B/cycle；支持 L1→LDS 直达 |
| XCD L2 | 4 MB/XCD，16-way | 16 channel；每 channel 每周期读 128 B、写 64 B |
| Infinity Cache | 256 MB/整卡，16-way | 位于 IOD，8 个 HBM stack 各对应 32 MB |
| HBM3E | 288 GB，8 stack | 8 Gbps pin rate；8 TB/s 峰值带宽 |

LDS 是软件可控制的数据复用区，相当于 CUDA 语境下的 shared memory；L1/L2/Infinity Cache 则主要由缓存策略和访问局部性决定。对 GEMM 而言，kernel 作者最直接控制的是 global load 的合并、LDS tile 布局、bank conflict、双缓冲和寄存器占用。

### 4.5 分区与 NUMA

计算可沿 XCD 边界分为：

- SPX：1 个 partition，每份 8 XCD；
- DPX：2 个 partition，每份 4 XCD；
- QPX：4 个 partition，每份 2 XCD；
- CPX：8 个 partition，每份 1 XCD。

内存模式为：

- NPS1：288 GB 在 8 个 HBM stack 上交织，编程简单，适合访问均匀的工作负载；
- NPS2：每个 IOD 形成 144 GB NUMA 域，流量留在本地 IOD，可减少跨 IOD 延迟、带宽和功耗成本。

DPX + NPS2 是自然配对：一个 4-XCD compute partition 对应一个 144 GB IOD 内存域。性能前提是分配和调度具有 NUMA affinity；若数据落在另一 IOD，仍要跨片上互联。

### 4.6 节点内互联

每条 x16 Infinity Fabric link 运行在 38.4 Gbps，单方向带宽为：

$$
16 \times 38.4\ \mathrm{Gb/s} \div 8 = 76.8\ \mathrm{GB/s}
$$

全双工合计为 153.6 GB/s。8 条物理链路中，典型 8-GPU 节点使用 7 条连接其他 GPU，1 条以 PCIe Gen5 连接 host/I/O。因此 P2P 单向峰值聚合为：

$$
7 \times 76.8 = 537.6\ \mathrm{GB/s}
$$

按双向 transport rate 口径则为：

$$
7 \times 153.6 = 1075.2\ \mathrm{GB/s}
$$

必须注意厂商表格中的 1,075.2 GB/s 是**双向聚合口径**，不能直接当作某次单向 all-reduce 的有效带宽。

## 5. AMD 与 NVIDIA 概念对照

先给结论：AMD 和 NVIDIA 都采用“线程层级 + 片上显式共享存储 + 矩阵乘加单元 + 多级缓存 + HBM”的基本组织，但命名、执行宽度、指令粒度和缓存拓扑并不相同。下面的对应关系适合迁移 mental model，不适合用来直接比较“核数”。

### 5.1 快速对照表

| AMD CDNA / ROCm | NVIDIA GPU / CUDA | 对应程度 | 说明 |
|---|---|---|---|
| GPU / device | GPU / device | 基本一致 | 一张加速卡或一个逻辑设备 |
| XCD（Accelerator Complex Die） | 没有严格对应；可粗略联想到 GPC/chiplet 计算分区 | 弱对应 | XCD 是物理 compute chiplet，自带 CU 和独立 L2；GPC 是片上图形/计算集群，不等同于独立 die |
| CU（Compute Unit） | SM（Streaming Multiprocessor） | 强对应 | 都是调度线程块、包含寄存器/L1/共享存储和执行管线的主要计算单元 |
| SIMD / vector ALU | CUDA Core / FP-INT pipeline | 功能对应 | 执行普通逐元素、地址及控制相关指令；数量口径不同，不能直接比较 |
| Matrix Core | Tensor Core | 强对应 | 都加速矩阵乘加；支持 dtype、指令 shape、线程协作粒度不同 |
| MFMA 指令 | `mma` / `mma.sync` / WGMMA | 语义对应 | 都计算小块 $D=AB+C$；MFMA 通常由 AMD wavefront 协作，NVIDIA 不同代可能由 warp 或 warpgroup 协作 |
| wavefront | warp | 强对应 | AMD CDNA 通常为 wave64，即 64 个 work-item；NVIDIA warp 固定为 32 个 thread |
| work-item | CUDA thread | 基本一致 | 一个逻辑线程 |
| workgroup | thread block / CTA | 基本一致 | 一组能同步并共享片上存储的线程 |
| grid / kernel dispatch | grid / kernel launch | 基本一致 | 所有 workgroup/thread block 的集合 |
| VGPR | 每线程 vector/general register | 强对应 | 保存各 lane 私有数据、地址和矩阵累加 fragment |
| SGPR | uniform/scalar register | AMD 特有程度较高 | 保存 wave 内一致的标量；NVIDIA 编程模型没有完全对称、显式的 SGPR 概念 |
| LDS（Local Data Share） | shared memory | 强对应 | workgroup/CTA 显式管理的片上 scratchpad；都要处理容量、bank conflict 和同步 |
| L1 vector data cache | L1 data cache | 强对应 | 每 CU/SM 附近的硬件缓存，但与 LDS/shared memory 的物理组织方式可能不同 |
| 每 XCD 4 MB L2 | 全 GPU shared L2 | 只在功能上对应 | AMD CDNA 4 的 L2 按 XCD 分布；NVIDIA 常把 L2 描述为全 GPU 共享的分区缓存 |
| Infinity Cache | NVIDIA 没有同名的一一对应层 | 无严格对应 | AMD 位于 IOD 的 256 MB memory-side cache；不能简单再称为 NVIDIA L2，因为 AMD 已有 XCD L2 |
| HBM3E | HBM3/HBM3E | 基本一致 | GPU 的高带宽全局内存 |
| Infinity Fabric（封装内） | NVIDIA 片上/封装内 interconnect | 功能对应 | 连接 XCD、IOD、cache 和 memory controller |
| Infinity Fabric Link（GPU 间） | NVLink | 强对应 | 节点内 GPU P2P 高速互联 |
| 8-GPU 全连接 IF 拓扑 | NVLink/NVSwitch 域 | 目标对应、拓扑不同 | 都服务 collective 和 P2P；CDNA 4 节点图是 GPU 间直连，NVSwitch 是交换结构 |
| ACE / HWS | SM work distributor / GPU scheduler | 弱对应 | 都把工作派发到计算单元，但硬件层级和公开模型不同 |
| SPX/DPX/QPX/CPX | MIG | 目标相似、机制不同 | 都可做资源隔离；AMD 沿 XCD 分区并配合 NPS，NVIDIA MIG 按 GPU instance 组合计算、缓存和内存资源 |
| NPS1/NPS2 | GPU 内 NUMA/内存亲和概念 | 无直接产品名对应 | CDNA 4 可把 HBM 分成两个 IOD 本地域；关键是计算分区和内存分配保持 affinity |
| HIP | CUDA Runtime / CUDA C++ | 强对应 | kernel 编程与运行时 API |
| ROCm | CUDA software platform | 强对应 | 驱动、编译器、库、工具和框架集成的总称 |
| `hipcc` / LLVM AMDGPU | `nvcc` / NVVM | 强对应 | GPU 编译工具链 |
| rocBLAS / hipBLASLt | cuBLAS / cuBLASLt | 强对应 | BLAS 与可调优矩阵乘库 |
| MIOpen | cuDNN | 强对应 | 深度学习算子库 |
| RCCL | NCCL | 强对应 | 多 GPU collective 通信库 |
| rocprofiler / Omniperf | Nsight Compute / Nsight Systems | 功能对应 | kernel 与系统性能分析工具，具体能力和指标名不同 |

### 5.2 最重要的层级映射：CU ≈ SM，wavefront ≈ warp

如果熟悉 CUDA，可以先使用以下翻译：

```text
AMD:    grid → workgroup → wavefront → work-item
NVIDIA: grid → thread block/CTA → warp → thread
```

其中最重要的差异是执行宽度：CDNA 的 wavefront 通常包含 64 个 work-item，而 NVIDIA warp 包含 32 个 thread。因此，把一个 CUDA warp-level 算法迁移到 AMD 时，不能只替换 API 名称，还必须检查：

- lane id 范围是 0–31 还是 0–63；
- shuffle、ballot 和 reduction 的 mask/width；
- 一个 workgroup 内需要多少个 wave；
- 分支发散时被屏蔽 lane 的比例；
- 矩阵指令的 fragment 在 64 个 lane 间如何分布。

CU 和 SM 都是 occupancy 的主要落点。一个 CU/SM 能同时驻留多少 workgroup/CTA，通常共同受以下资源约束：

$$
N_{resident} = \min
\left(
N_{thread},
N_{register},
N_{shared\ memory},
N_{hardware}
\right)
$$

在 AMD 上，`shared memory` 对应 LDS，寄存器还要区分 VGPR 与 SGPR。CDNA 4 将 LDS 扩到 160 KB 后，LDS 容量约束可能减轻，但如果 tile 同时增加 VGPR 累加器，occupancy 仍可能被 VGPR 限制。

### 5.3 Matrix Core ≈ Tensor Core，但“一个 core”不可直接比较

两者的共同语义都是：多个线程协作执行一个小矩阵乘加，而不是每个线程各自做完整矩阵乘法。

$$
D_{m\times n}=A_{m\times k}B_{k\times n}+C_{m\times n}
$$

在 AMD CDNA 上，kernel 通常通过 MFMA 指令使用 Matrix Core；在 NVIDIA 上，对应的是 `mma.sync`、Hopper WGMMA 等指令。它们的差异包括：

- 指令的 $m,n,k$ shape；
- 由一个 wave64、warp32 还是 warpgroup 协作；
- A/B fragment 来自 VGPR、LDS/shared memory 还是异步搬运路径；
- 支持 FP64、TF32、BF16、FP8、MXFP 等哪些输入格式；
- accumulator 类型和每 lane 持有的元素布局；
- 每周期能发射多少条指令以及 pipeline latency。

因此“MI355X 有 1,024 个 Matrix Core”和“某 NVIDIA GPU 有多少 Tensor Core”不能按数量相除得出性能比例。可靠比较必须统一 dtype、dense/sparse 口径后比较整卡峰值，并最终查看同一 kernel 的实测吞吐。

### 5.4 LDS ≈ shared memory

这是 kernel 迁移中最直接的对应关系：

```cpp
// CUDA
__shared__ half tile[...];

// HIP 源码通常仍可使用相似语法
__shared__ half tile[...];
```

两者都是由 thread block/workgroup 中所有线程共享、由软件显式寻址的片上存储，常用于：

- 合并 global memory load；
- 对 A/B tile 做多次复用；
- 改变 layout 或转置；
- wave/warp 间交换数据；
- producer-consumer 双缓冲。

但不能假设 bank 数、bank 宽、广播行为和冲突规则完全一样。一个在 NVIDIA shared memory 上无 bank conflict 的 swizzle，迁移到 AMD LDS 后必须重新测量。CDNA 4 的特殊增强是 160 KB LDS、256 B/cycle 读带宽及 L1→LDS 直接加载；它在概念上类似 NVIDIA 通过异步 global/shared-memory copy 路径减少寄存器中转，但两者不是同一条 ISA 机制。

### 5.5 缓存层次不能机械对齐

可以用下面两条简化路径建立直觉：

```text
AMD CDNA 4:
HBM → Infinity Cache（整卡/IOD）→ XCD L2 → CU L1/LDS → registers

NVIDIA（概念化）：
HBM → shared L2 → SM L1/shared memory → registers
```

关键差别是 CDNA 4 同时存在每 XCD 4 MB L2 和 IOD 上的 256 MB Infinity Cache。把 Infinity Cache 直接翻译成 NVIDIA L2 会遗漏 XCD L2；把 XCD L2 直接当作全卡共享 L2，又会遗漏它的 chiplet locality。

这会影响 kernel 和系统优化：

- 同一 XCD 内的 producer/consumer 更可能复用本地 L2；
- 跨 XCD 访问需要经过片上 Infinity Fabric；
- NPS2 下还应让 XCD 访问所属 IOD 的本地 HBM；
- NVIDIA 上熟悉的 L2 persistence、CTA cluster 或 TMA 优化不能假定在 AMD 上存在同构实现。

### 5.6 Infinity Fabric ≈ NVLink，但要区分封装内与 GPU 间

`Infinity Fabric` 是一组互联技术的总称，在本文里至少有两种语境：

1. **GPU 封装内部**：连接 XCD、IOD、Infinity Cache 和 HBM 控制器；更像 NVIDIA GPU 内部的 NoC/片上互联。
2. **GPU 封装之间**：Infinity Fabric Link 连接节点内其他 GPU；这一层才最接近 NVLink。

CDNA 4 的 8-GPU 节点采用每 GPU 到其余 7 张 GPU 的直接连接。NVLink 系统则可能通过 NVSwitch 构建交换网络。虽然上层都运行 RCCL/NCCL collective，但路由、争用和双向聚合带宽口径不同，不能只比较链路宣传数字。

### 5.7 AMD 分区 ≈ NVIDIA MIG，但不是同一种切法

共同目标是把一张物理 GPU 隔离成多个可独立调度的实例。CDNA 4 的 compute partition 以 XCD 为粒度：CPX 每份 1 XCD，QPX 每份 2 XCD，DPX 每份 4 XCD，SPX 使用全部 8 XCD；内存再通过 NPS1/NPS2 决定交织或 IOD 本地化。

NVIDIA MIG 也隔离计算、缓存和内存资源，但实例规格和物理切分方式由具体 GPU 代际定义。迁移部署方案时，应比较每个实例实际获得的 CU/SM、HBM 容量和带宽、copy engine、互联能力及故障隔离，而不是只比较实例数量。

### 5.8 从 CUDA kernel 迁移到 AMD 时的检查清单

1. 把 SM/thread block/warp/thread mental model 映射到 CU/workgroup/wavefront/work-item。
2. 检查所有隐含 `warpSize == 32` 的代码；HIP 中应使用目标平台的 warp size，而非硬编码 32。
3. 将 shared-memory tile 迁移到 LDS 后，重新设计或验证 bank-conflict swizzle。
4. 不要假定 NVIDIA MMA fragment layout 与 AMD MFMA fragment layout 相同；应使用对应库或 ISA 定义。
5. 重新调 block/workgroup shape。相同线程数不代表相同 wave 数、occupancy 或调度效果。
6. 分别检查 VGPR、SGPR 和 LDS 使用量；CUDA 侧只看 register count 的经验不够。
7. 把 `cuBLASLt/cuDNN/NCCL` 路径分别映射到 `hipBLASLt/MIOpen/RCCL`，先用库实现建立基线。
8. 用 rocprofiler/Omniperf 实测 Matrix Core 利用率、HBM/L2/LDS 带宽和 wave occupancy，不照搬 Nsight 上的阈值。
9. 多 XCD 场景检查 partition 与 NPS affinity；这是单片 NVIDIA GPU mental model 中容易遗漏的一层。

## 6. 关键公式 / 算法

### 6.1 峰值算力如何得到

作用：把“每 CU 每周期操作数”连接到整卡 PFLOP/s，检查表格是否自洽。

$$
P_{peak} = N_{CU} \cdot r_{CU} \cdot f
$$

- $N_{CU}=256$：启用 CU 数；
- $r_{CU}$：每 CU 每周期 FLOP 数；
- $f$：engine clock，MI355X 峰值 2.4 GHz；
- $P_{peak}$：理论峰值 FLOP/s。

例如 FP16 Matrix 的 $r_{CU}=4096$：

$$
256 \times 4096 \times 2.4\times10^9
=2.5166\times10^{15}\ \mathrm{FLOP/s}
\approx 2.5\ \mathrm{PFLOP/s}
$$

MXFP4 的 $r_{CU}=16384$，因此：

$$
256 \times 16384 \times 2.4\times10^9
\approx 10.07\ \mathrm{PFLOP/s}
$$

这也证明 Table 1 的 `16834` 是笔误。峰值公式默认所有 CU 都持续满频、发射满速矩阵指令，没有访存、依赖、同步、调度或功耗限制，真实 kernel 必然更低。

PyTorch 语义只是普通矩阵乘；实际是否命中 Matrix Core 由 dtype、shape、layout 和后端 kernel 决定：

```python
# 语义层；并不代表实际硬件指令或量化路径
D = A @ B + C
```

### 6.2 Roofline：为什么 10 PFLOP/s 不代表 decode 快 4 倍

作用：判断 kernel 是被 Matrix Core 上限限制，还是被 HBM 带宽限制。

$$
P_{attainable} \le \min(P_{peak},\ I \cdot BW_{HBM})
$$

其中算术强度为：

$$
I = \frac{\text{FLOPs}}{\text{bytes transferred from HBM}}
$$

MI355X 对 FP8 dense Matrix 的机器平衡点约为：

$$
I^* = \frac{5.0\ \mathrm{PFLOP/s}}{8\ \mathrm{TB/s}}
\approx 625\ \mathrm{FLOP/byte}
$$

对 MXFP4/6 峰值则约为 1,250 FLOP/byte。算术强度低于这个阈值时，即使矩阵核空闲，性能也主要由 HBM 决定。小 batch decode 的矩阵向量乘和 KV cache 扫描通常复用不足，因此新增低精度算力更可能通过“更少字节/元素”而不是“更多 FLOP/cycle”带来收益。

### 6.3 KV cache 容量

对标准多头注意力，单请求 KV cache（不考虑对齐和分页元数据）约为：

$$
M_{KV}=2 \cdot L \cdot S \cdot H_{KV} \cdot D \cdot b
$$

- $L$：层数；
- $S$：上下文 token 数；
- $H_{KV}$：KV head 数；
- $D$：每 head 维度；
- $b$：每元素字节数；
- 系数 2 对应 K 和 V。

例如 $L=80$、$S=131072$、$H_{KV}=8$、$D=128$、BF16 的 $b=2$：

$$
M_{KV}=2\times80\times131072\times8\times128\times2
=40\ \mathrm{GiB}
$$

这说明 288 GB 的价值不仅是放权重，也可容纳更多并发请求或更长上下文。它不改变 KV cache 随 $S$ 线性增长这一算法事实；decode 仍需高效 paged KV、continuous batching 和 attention kernel。

## 7. 数据流和实现设计

### 7.1 GEMM kernel 映射

以 $A[M,K]B[K,N]$ 为例，一个高性能 HIP/Triton kernel 通常需要：

1. 将输出划分为 $B_M\times B_N$ workgroup tile；
2. 沿 K 维循环，每轮从 HBM/L2 加载 $B_M\times B_K$ 的 A tile 和 $B_K\times B_N$ 的 B tile；
3. 在 160 KB LDS 中做 staging、swizzle 和双缓冲；
4. wavefront 从 LDS 读 fragment，发出矩阵乘加；
5. 累加器驻留 VGPR，最后做 epilogue 并写回。

CDNA 4 对该流程的直接影响：

- 更大 LDS 允许更大的 tile、更深的 pipeline 或更多 operand/scale 缓存；
- 256 B/cycle LDS read 提高向 Matrix Core 供数能力；
- L1→LDS 直达减少 VGPR staging，可能降低寄存器压力并改善 occupancy；
- 更宽的低精度 Matrix Core 要求 kernel 增加并行 K 工作或数据复用，否则更容易被供数限制；
- MX 格式还需使 32-value scale block 与 tile/layout 对齐，避免 scale gather 和转换吞掉收益。

160 KB 是物理容量，不等于单个 workgroup 可以无条件独占全部空间。实际 tile 仍受 workgroup 数、VGPR、LDS 分配粒度和目标 occupancy 约束；白皮书没有披露这些限制。

### 7.2 Prefill 与 decode

**Prefill** 通常有较大的 $M$ 或序列维，GEMM 与 attention tile 有较高复用，较容易接近 Matrix Core 吞吐。更大 LDS、FP8/MXFP 和双倍超越函数吞吐都能直接受益。FlashAttention 类 kernel 可让 Q/K/V block 驻留 LDS/寄存器，避免将完整 $QK^T$ 写回 HBM。

**Decode** 每步 token 数少，权重和 KV cache 读取占比高，经常 memory-bound。此时：

- 8 TB/s HBM 和低 bit 权重/KV 比 10 PFLOP/s 数字更关键；
- 288 GB 提高 batch、上下文和模型容量；
- continuous batching 可把多个请求拼成更大的矩阵，提高计算利用率；
- paged KV cache 仍需软件实现，白皮书没有描述硬件级分页 attention；
- GPU partition 可用于隔离多个小模型，但分区会减少单实例可用 XCD 和内存域。

### 7.3 Cache 与数据复用

- L1 是每 CU 私有，适合局部数据；LDS 显式管理，适合确定性 tile 复用。
- 4 MB L2 是每 XCD 独立，跨 XCD 共享数据需要走 Infinity Fabric/Infinity Cache，不能把整卡看作统一 32 MB L2。
- 256 MB Infinity Cache 可吸收跨 XCD 或 HBM 热数据，但大型模型权重远超其容量，不能替代 HBM 带宽优化。
- L2 的 writeback + write-allocate 以及“dirty writeback 后保留副本”可减少重复填充；对实际 kernel 的收益仍取决于访问模式。

### 7.4 多 GPU 与分布式训练/推理

8 GPU 全连接避免节点内多跳路由，但 collective 的有效带宽还受消息大小、算法、链路并发、同步和软件栈影响。适合关注：

- tensor parallel 的 all-reduce / reduce-scatter；
- expert parallel 的 all-to-all；
- pipeline parallel 的点对点激活传输；
- KV/weight sharding 与 NUMA/partition affinity。

白皮书给出的 >1 TB/s 是聚合理论 transport rate，不是 RCCL collective 实测。评估真实系统必须用对应消息大小和拓扑下的 RCCL benchmark。

### 7.5 软件落地

官方定位的上层栈包括 PyTorch、JAX、Megatron-LM、TorchTitan、MaxText、vLLM 和 SGLang；kernel 层包括 ROCm 库、profiler 与 Triton。对开发者而言，建议按以下顺序落地：

```text
框架/serving 可运行
  → rocBLAS/hipBLASLt/attention 库命中正确 kernel
  → profiler 判断 compute-bound / bandwidth-bound / communication-bound
  → 调整 dtype、batch、layout、并行策略和 NUMA affinity
  → 只有库 kernel 不满足时再写 HIP/Triton 自定义 kernel
```

白皮书宣称支持 FlashAttention v3、Sliding Window Attention 及热门模型的 Day-0 优化，但没有给出版本、kernel 路径或结果，部署时需要在目标 ROCm 版本上单独验证。

## 8. 实验和效果

这份白皮书没有传统论文意义上的实验，主要是规格和 AMD 内部计算/测量结果。

### 8.1 可信度较高的结构性结论

- 256 CU、1,024 Matrix Core、288 GB HBM3E、8 TB/s 等属于产品规格；
- 每 CU FP16/FP8 操作数翻倍与峰值公式相互自洽；
- LDS 160 KB 和 256 B/cycle 是明确的微架构声明；
- 2 IOD、8 XCD、8 HBM stack 的拓扑由多张结构图交叉支持。

### 8.2 需要谨慎解释的结果

- “1.9×”低精度整卡峰值来自单 CU 2×吞吐，但 CU 数从 304 降至 256，频率也不同，所以整卡不是精确 2×。
- “3.9× FP4/FP8”比较的是 CDNA 4 新 MXFP4 峰值与上一代 FP8 峰值，**格式不同**，既包含每 CU 资源变化，也包含每元素 bit 数变化。
- “7.7× 最优 partition”同时改变了代际、partition 粒度和比较 dtype（CDNA 4 FP4 对 CDNA 3 FP8），不应理解成同一 workload 的 7.7× 加速。
- “118% rack compute 提升”依赖 200 kW 液冷机架能放 128 张 MI355X，而比较对象是较低功率/密度配置，主要是系统密度而非单卡加速。
- 稀疏峰值默认符合硬件要求的结构化稀疏。未经剪枝和精度恢复的 dense 模型不能直接获得 2×。

### 8.3 缺失实验

- 无 GEMM/attention 实测利用率；
- 无训练 tokens/s、推理 TTFT/TPOT/throughput；
- 无 FP8/MXFP4 精度对比和量化 recipe；
- 无 1–8 GPU scaling efficiency 或 RCCL collective 曲线；
- 无 MI325X 与 MI355X 的同功耗、同 dtype、同模型比较；
- 无 NPS1/NPS2 的 workload latency 和带宽数据；
- 无 1,000 W 与 1,400 W 下的性能/瓦。

因此，这份材料适合建立架构 mental model 和优化假设，不足以单独支持采购或模型性能预测。

## 9. 我的判断

### 真正贡献

CDNA 4 的价值在于一次较完整的**系统再平衡**：先进制程资源投入低精度 Matrix Core，LDS 容量和带宽同步扩张，HBM3E 带宽/容量提高，IOD 拓扑简化。这比只看“10 PFLOP/s”更重要，因为任何单点扩张都会把瓶颈推到下一层。

### 值得精读

- Figure 5：完整封装、缓存、HBM 和外部互联数据路径；
- Table 1：每 CU 每周期吞吐，能反推真实架构取舍；
- 第 9 页：LDS/L1/L2 的具体组织，是 kernel 优化最有用的部分；
- 第 11–13 页：partition 与 NPS1/NPS2，影响部署和 NUMA affinity；
- Endnotes：揭示宣传数字使用的比较口径。

### 可以快速跳过

- 通用 AI 历史背景；
- 机架产品形态和 Kubernetes 宣传；
- 未附 benchmark 的 Day-0 模型支持列表。

### 价值评级

| 方向 | 评级 | 原因 |
|---|---|---|
| 算法理解 | 低 | 没有新算法或数学方法 |
| 推理系统 | 中-高 | 容量、带宽、分区、互联直接影响 serving 设计 |
| kernel/算子优化 | 高 | LDS、缓存、低精度吞吐和数据格式决定 GEMM/attention 映射 |
| 是否值得“复现” | 中 | 架构无法复现；值得做针对性的 microbenchmark 和 roofline 验证 |

### 关键勘误与疑点

1. 文中标题多次写成 `CNDA 4`，应为 `CDNA 4`。
2. Table 1 的 MXFP6/MXFP4 `16834 FLOPS/clock/CU` 应为 `16,384`。
3. 结论页写“over 10 TFLOP/s”，结合 Table 2 应为 **10 PFLOP/s**。
4. 引言处“10 TFLOP/s peak theoretical for MXFP6/MXFP4”同样应为 **10 PFLOP/s**。
5. Table 1 把 MI325X 列标题写成 MI300X，虽然正文称比较 MI355X 与 MI325X；两者核心峰值接近，但引用时应标明原表不一致。
6. 白皮书说 Transformer 常用的 activation 是 softmax，这个表述不严谨：softmax 是 attention 概率归一化；FFN 常用 GELU/SwiGLU 等激活。

## 10. 下一步阅读建议

1. 先重看 Figure 5，并能不看图画出 `HBM → IOD/Infinity Cache → Fabric → XCD/L2 → CU/L1/LDS → Matrix Core`。
2. 再读 Table 1，用峰值公式自行验证 FP16、FP8、MXFP4 三行，建立“每 CU 吞吐—频率—整卡峰值”的对应关系。
3. 深挖第 9 页 LDS：后续应查 CDNA 4 ISA/ROCm 文档，确认 MFMA 指令 shape、operand/accumulator dtype、LDS bank 数与宽度、L1→LDS 指令语义。
4. 做三个 microbenchmark：大 GEMM 测 compute ceiling、stream copy 测 HBM ceiling、LDS bandwidth/bank-conflict 测片上供数；再画实测 roofline。
5. 对推理系统，分别测 prefill 与 decode，并固定模型、dtype、batch、context，对比 NPS1/NPS2、单分区/多分区以及 vLLM/SGLang 的 TTFT、TPOT、吞吐。
6. 对多 GPU，使用 RCCL 测 all-reduce、all-gather、reduce-scatter 和 all-to-all，不能用 1,075.2 GB/s 理论聚合值代替 collective 性能。
7. 若要研究 MXFP4/MXFP6，下一步应优先读 OCP MX 规范和 ROCm 对应类型/矩阵指令文档，再看框架量化 recipe；白皮书本身不足以实现正确的量化 kernel。
