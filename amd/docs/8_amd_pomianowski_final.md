# AMD RDNA 4 计算架构摘录

> 原文：[8_amd_pomianowski_final.pdf](../pdfs/8_amd_pomianowski_final.pdf)  
> 来源：Hot Chips 2025 演讲，Andy Pomianowski、Laks Pappu。  
> 范围：本文只保留通用计算相关内容：SIMD/wave、寄存器、LDS/缓存、矩阵计算、内存系统和 SoC 数据流；省略光栅化、光线追踪、显示、多媒体和游戏性能部分。

## 论文一句话总结

从计算视角看，RDNA 4 是一个以 wave32 SIMD 为基础、加入更强低精度矩阵单元，并通过 128 KB LDS、动态 VGPR、乱序内存返回和压缩数据路径改善利用率的 GPU 架构。

它仍是游戏 GPU，不是专为 AI 训练设计的 CDNA；但其计算路径已经能直接执行通用 compute shader 和 WMMA 类矩阵乘加，适合本地推理、图像/视频处理、物理模拟和其他高并行任务。

## 1. 背景和问题

GPU 通用计算的难点不只是 ALU 数量。一个 kernel 还会受寄存器占用、LDS 容量与 bank 冲突、缓存/显存带宽、数据布局、分支和长延迟 load 限制。尤其是低精度矩阵计算，若 tile 布局不匹配或数据不能留在寄存器/LDS 中复用，矩阵单元很快会因等数据而闲置。RDNA 4 的计算侧目标是提高每 CU 的有效吞吐，并为更多低精度矩阵工作提供硬件路径。

## 2. 核心结论

- 基本执行模型仍是 **wave32**：32 个线程在一个 SIMD32 上执行同一条指令、处理不同数据。一个 kernel 可以有 64、128、256 等更多线程，硬件将其拆为多个 wave32。
- 一个 Compute Engine/Dual-CU 级块含四个 SIMD32、四组 scheduler/vector register 资源，并让相邻 CU 共享 128 KB LDS。PPT 在不同页面混用 `Compute Engine`、`Dual Compute Unit` 和 `WGP`，此处将其视为两个 CU 组成的共享资源执行块。
- 每个 SIMD32 有 32 个 FP FMA ALU、32 个 FMA/INT ALU 和 8 个 transcendental logic unit（TLU）；scalar 单元处理 wave 内一致的控制和地址计算。
- RDNA 4 加强 Matrix Accelerator / WMMA，支持 16/8/4-bit tensor input 与 16/32-bit accumulator，并加入 FP8 和 4:2 structured sparsity 支持。
- 每个 vector 域有 192 KB VGPR、8 KB SGPR；动态 VGPR 分配允许一个 shader 在不同阶段按需申请/归还寄存器，以提高可驻留 wave 数，但需要软件处理资源不足时的等待。
- 数据可经 32 KB L0、128 KB LDS、CU aggregate cache、8 MB L2、64 MB Infinity Cache 和 GDDR6 层级流动；集中式硬件压缩/解压旨在减少 fabric 与显存侧实际搬运的字节数。
- RDNA 4 支持跨不同 shader 的乱序 memory return：后发且较快完成的请求可不必等待先发的长延迟请求，从而减少 head-of-line blocking、提高可执行 wave 比例。

## 3. 形象解释 / Mental Model

### 3.1 GPU 计算像许多 32 人小队轮流使用工厂

可以把一个 compute kernel 看成一大批相同任务，例如对数组的每一项做计算。GPU 把线程分为 32 人小队，即 wave32；每队在 SIMD32 上同步执行相同指令，但每个人读取自己的数据。

```text
kernel 的 256 个线程
  → 8 个 wave32
  → 分派到多个 SIMD32
  → 每条 SIMD32 执行一条向量指令
  → scheduler 在等待内存的 wave 与可执行 wave 之间切换
```

理想情况是一个 wave 内 32 个线程走同一控制路径且连续访问数据。若 16 个线程走 `if`、16 个走 `else`，硬件仍会正确执行两条路径，但每条路径会有一半 lane 空闲；若所有 wave 都在等 GDDR6，SIMD 也会空闲。因此高性能不仅是“有多少 ALU”，而是“是否有足够可执行 wave 和足够快的数据供给”。

### 3.2 Compute Engine 是一个“通用车间 + 矩阵专机 + 共享周转区”

从 PPT 第 5 页可将一个 Compute Engine 理解为两个相邻 CU 形成的执行块：四条 SIMD32 通用生产线做普通向量/整数/超越函数计算，Matrix Accelerator 做规则的低精度矩阵乘加，128 KB LDS 是软件可管理的共享周转区。

```text
4 × SIMD32：通用 vector / integer / transcendental 指令
        ↕
VGPR / SGPR：保存每线程临时值与 wave 共享状态
        ↕
Matrix Accelerator：小矩阵乘加
        ↕
128 KB LDS：work-group 显式复用的 tile / scratchpad
        ↕
L0 → L2 → Infinity Cache → GDDR6
```

真实对象分别是：VGPR 保存每个 lane 不同的值，SGPR 保存一个 wave 共享的值，LDS 保存一个 work-group 共享的数据 tile，cache 负责自动保留近期读取的数据。它们存的内容不同，不能互相替代。

### 3.3 一个矩阵 tile 怎么流动

考虑矩阵乘法：

$$
C_{m\times n} \leftarrow A_{m\times k}B_{k\times n} + C_{m\times n}
$$

一个 work-group 负责 $C$ 的一个 tile：

1. 各 wave 从全局内存读取 $A,B$ 的 tile；连续、合并的访问先经过 L0/L2/Infinity Cache。
2. 可复用 tile 放入 LDS，避免不同 wave 或 K-loop 重复从更远缓存/显存读取。
3. `load and transpose` 将内存布局整理为 Matrix Accelerator 所需的寄存器/lane 布局。
4. Matrix Accelerator 消费 FP16、FP8 或低位宽整数输入，累加结果保留在 16/32-bit accumulator/register 中。
5. SIMD32 执行矩阵单元不适合的索引、边界、激活、归一化或 elementwise 操作；K 维循环继续复用 LDS/寄存器中的数据。
6. 完成的 $C$ tile 经 cache hierarchy 写回全局内存。

如果只记一张图，应记住：**HBM/GDDR6 → cache → LDS/寄存器 → Matrix Accelerator/SIMD → 寄存器/LDS → cache → HBM/GDDR6**。矩阵单元只是中间一步；真正决定实际速度的是 tile 复用、寄存器压力和内存搬运。

## 4. 方法总览

### 4.1 执行层次与线程 shape

| 层次 | 含义 | 在 RDNA 4 中的作用 |
|---|---|---|
| kernel / compute shader | 软件写的一份并行程序 | 例如矩阵乘法、卷积、滤波或物理模拟 |
| work-group | 软件定义的一组可协作线程 | 共享 LDS、可使用 barrier |
| wave32 | 32 个同步执行的线程 | 一条 SIMD32 的原生执行粒度 |
| SIMD32 | 32-lane 向量执行硬件 | 对 32 个不同数据元素同时执行同一指令 |
| CU | 调度、寄存器、SIMD、LDS/cache 接口等资源的计算模块 | 驻留并运行多个 wave32 |
| Dual CU / Compute Engine / WGP | 两个 CU 的共享资源执行块 | PPT 图中拥有 4 个 SIMD32 和共享 128 KB LDS |

对普通 elementwise kernel，可把数据写成：

```python
# x, y, out: [N]
i = global_thread_id()
out[i] = x[i] * scale + y[i]
```

每个 wave32 处理连续或近连续的 32 个 $i$。这时 32 lane 都有效，且读取通常可合并为少数 cache line transaction。线程总数不必是 32，但 work-group size 最好是 32 的倍数，避免最后一个 wave 留下许多无效 lane。

### 4.2 通用向量与 scalar 路径

PPT 中每个 SIMD32 具有：

| 单元 | 数量 | 适合的指令 |
|---|---:|---|
| FP FMA ALU | 32 | FP32/相关浮点向量乘加 |
| FMA/INT ALU | 32 | 混合浮点乘加或整数算术/地址相关计算 |
| TLU | 8 | exp、log、sin 等较复杂数学函数 |
| scalar unit | 每个 scheduler 域一套 | wave 内一致的地址、分支、循环计数、常量 |

scalar/vector 分工可以抽象成：

$$
a \leftarrow g(a), \qquad
\mathbf{y} \leftarrow \operatorname{FMA}(\mathbf{x},a,\mathbf{b})
$$

其中 $a$ 是一个 wave 共享的 scalar，$\mathbf{x},\mathbf{b},\mathbf{y}\in\mathbb{R}^{32}$ 是 32 lane 的值。把 uniform 工作放入 scalar 路径可避免 32 个 lane 重复计算；把真正不同的数据留在 vector 路径。

### 4.3 矩阵路径与低精度

Matrix Accelerator 的数据通路是：

```text
load → transpose/repack → register storage → WMMA → 16/32-bit accumulator
```

它接受 16/8/4-bit tensor input，矩阵核心本身有 64 个计算单元，结果以 16 或 32 bit 累加。16-bit 或更小的输入减少每个元素的访存量并提高单周期可完成的乘加数；较宽 accumulator 降低长 K 维累加的数值误差。

PPT 的相对峰值图以 RDNA 3 的 FP16 为 1，并注明 **RDNA 4 数字包含稀疏性**。图中 RDNA 4 的 FP16 为约 4、FP8/INT8/INT4 为约 8 的归一化上限；其中 4:2 structured sparsity 最多可额外带来 2 倍峰值。因此这些数字不能当作纯 dense throughput：模型权重/激活必须满足支持的稀疏模式，且 kernel 要处理 metadata、tile 对齐、量化/反量化和累加精度。

### 4.4 AMD RDNA 4 与 NVIDIA CUDA 术语对照

下表按功能对照，**不是晶体管级的一对一等价**。尤其 `CU` 与 `SM` 的粒度不同，具体缓存、调度器数量和每周期发射能力还会随 NVIDIA 架构代际变化。

| AMD RDNA 4 | NVIDIA CUDA 语境中最接近的概念 | 关系与区别 |
|---|---|---|
| kernel / compute shader | CUDA kernel | 都是一份并行程序，由大量线程执行 |
| work-group | thread block | 都是可协作、可同步、共享片上 scratchpad 的线程组 |
| wave32 | warp | 最接近的一对：均为 32 个锁步线程 |
| SIMD32 | 一个 warp 的执行分区（warp scheduler + 32-lane execution path） | 都执行一个 32 线程组的同一条指令；NVIDIA 通常以 warp scheduler/执行分区描述，不把它称为 SIMD32 |
| CU | SM 的一个子分区，近似而非等价 | 一个 RDNA CU 通常有两个 SIMD32；不能将一个 CU 直接等同一个完整 SM |
| Dual CU / Compute Engine / WGP | Streaming Multiprocessor（SM） | 都是共享 LDS/shared memory、cache 和调度资源的更大执行域；这是最有用的近似层级 |
| VGPR | per-thread register file | AMD 一个 VGPR 表示 wave 中每 lane 一份值；CUDA 源码中一个局部标量通常映射为每线程一个 register |
| SGPR + Scalar Unit | 无公开的一对一对应 | NVIDIA 也会优化 warp-uniform 值和常量访问，但公开 CUDA 模型不暴露独立 SGPR/Scalar Unit 这种对应资源 |
| LDS | shared memory | 最接近：均为 block/work-group 显式管理的片上 SRAM，需处理容量、同步和 bank conflict |
| L0/L1 vector/texture cache | L1 data/texture cache | 都用于临近 load/store/texture 的自动缓存；具体合并方式和一致性规则依架构而异 |
| L2 / Infinity Cache | L2 cache | 都为更大范围的自动缓存；Infinity Cache 是 AMD 在 L2 之外加入的大容量缓存层，不能简单当成 NVIDIA 某一层固定 cache |
| Matrix Accelerator / WMMA | Tensor Core / WMMA 或 MMA 指令 | 都执行低精度矩阵乘加；tile shape、支持格式、稀疏条件、寄存器 layout 不同 |

最简记忆方式是：

```text
AMD：work-group → wave32 → SIMD32 → CU → Dual CU/WGP
NVIDIA：thread block → warp → execution partition → SM
```

其中 `wave32 ≈ warp` 最准确；`Dual CU/WGP ≈ SM` 最方便；`CU ≈ ?` 则没有可靠的一对一等号。写跨平台 kernel 时，应根据实际的 wave/warp size、register、shared memory/LDS、cache 和矩阵指令重新调优，不能把某一平台的 tile 或 occupancy 参数原样搬到另一平台。

## 5. 关键机制

### 5.1 LDS：软件显式控制的数据复用区

LDS 是 work-group 共享的片上 SRAM，可类比 CUDA 的 shared memory。两个相邻 CU 形成的 Compute Engine 共享 128 KB LDS；它比自动 cache 更可控，但需要 kernel 明确搬运、同步和布局。

对 tile 计算，访问次数可抽象为：

$$
\text{HBM reads without reuse} \propto mkn,
\qquad
\text{HBM reads with tiled reuse} \ll mkn
$$

这不是精确复杂度式，而是强调：如果每次 FMA 都回显存取 $A,B$，计算会立即 memory-bound；把 tile 搬入 LDS 后，一个元素可被同一 work-group 的多个 lane、多轮 K-loop 复用。

LDS 的风险是 bank conflict、barrier 开销和过大分配。某个 work-group 占用越多 LDS，能同时驻留的 work-group 越少；因此 tile 不能无限做大。

### 5.2 动态 VGPR：按阶段分配寄存器

静态寄存器分配按整个 shader 的最坏情况 $R_{\max}$ 预留，因此可驻留 wave 数受限于：

$$
N_{\text{waves}} \leq
\left\lfloor\frac{R_{\text{VGPR,total}}}{R_{\max}\cdot 32}\right\rfloor
$$

许多计算程序的临时变量并非全程同时活跃。例如一个阶段只做地址/索引，下一阶段才需要一大块临时 tile。RDNA 4 允许在运行中申请和归还部分 VGPR，使低需求阶段可驻留更多 wave，帮助隐藏 memory latency。

这不是自动减少寄存器数量：软件仍要指定何时申请，资源不足时必须等待。对 kernel 作者而言，基础策略仍是控制局部数组、循环展开和中间 tensor 的活跃范围，先降低 $R_{\max}$。

### 5.3 乱序 memory return：避免队头阻塞

若先发出的请求 $r_1$ 是长延迟 miss，后发的 $r_2$ 是短延迟 hit，严格返回顺序会让 $r_2$ 被 $r_1$ 挡住。RDNA 4 对不同 shader 的请求允许更接近完成时间的乱序返回：

$$
\operatorname{return\ order} \approx \operatorname{sort}(t_i)
$$

其中 $t_i$ 是请求的实际完成时间。收益是：已经拿到数据的 shader wave 可以先恢复执行，不必等无关 wave 的远端访问。它不打破单个线程/单个 shader 内的数据依赖，正确性仍由硬件依赖追踪保证。

## 6. 数据流和实现设计

### 6.1 Cache、Infinity Fabric 与 GDDR6

计算数据的主要路径为：

```text
SIMD / Matrix Accelerator
  ↕ VGPR、SGPR、LDS
L0 (32 KB)
  ↓
CU aggregate cache
  ↓
L2 (8 MB)
  ↓
Infinity Cache (64 MB)
  ↓
Infinity Fabric
  ↓
memory controller → GDDR6
```

PPT 给出 Infinity Fabric 为 coherent、1 KB/clock、1.5–2.5 GHz。RX 9070 XT 配置为 256-bit、最高 20 Gbps GDDR6 和 16 GB 容量。对 compute kernel，L0/L2/Infinity Cache 命中率决定实际读取是否需要走到更远、更耗电的 GDDR6；高矩阵峰值并不能替代足够的 cache reuse。

### 6.2 集中式压缩/解压

RDNA 4 将压缩/解压逻辑放在 SoC 数据通路中，而非要求每个计算客户端自行知道数据是否压缩。抽象流程为：

```text
producer 写逻辑数据
  → 硬件压缩并记录 metadata
  → Infinity Fabric / memory 传输压缩数据
  → consumer 读取时硬件解压为逻辑数据
```

它的收益来自减少实际移动的 byte 数，不是减少算法所需的 logical element 数。PPT 给出的约 25% fabric bandwidth 降低只针对少数 workload，计算 kernel 是否受益取决于其数据类型、可压缩性和读写模式。

### 6.3 计算 kernel 的实际瓶颈检查表

| 现象 | 优先检查 | 常见改进方向 |
|---|---|---|
| SIMD 利用率低 | wave 内分歧、非 32 倍数线程数 | 重排数据/分支；让 block size 对齐 wave32 |
| 矩阵单元利用率低 | load/repack、VGPR、LDS、HBM 带宽 | 增大合理 tile、双缓冲、提高 K 维复用 |
| occupancy 低 | VGPR/LDS 分配、过度 unroll | 缩小临时数组、控制 unroll、分阶段释放资源 |
| 等内存时间长 | cache miss、访问跨步、请求依赖 | 合并访问、提升局部性、增加独立 load 与并发 wave |
| 带宽功耗高 | 可压缩性、写回量、重复读取 | 使用合适数据格式、复用 tile、减少无效写回 |

## 7. 实验和效果

PPT 对计算部分主要给出架构能力，不给出可复现的通用 compute benchmark。可谨慎采用的数字只有：

- WMMA 支持 FP16、FP8、INT8、INT4 等输入，以及 16/32-bit accumulate；
- 4:2 structured sparsity 最多可将适用格式的峰值再提高 2 倍；
- 相对峰值图将 RDNA 3 FP16 归一化为 1，RDNA 4 图示上限为 FP16 约 4、FP8/INT8/INT4 约 8，但图明确包含 sparse rate；
- 64 MB Infinity Cache、8 MB L2、合计 2 MB CU cache、16 GB/256-bit GDDR6 是 RX 9070 XT 的资源配置。

缺失的关键信息包括 WMMA 指令 tile shape、每周期操作数、寄存器布局、LDS/矩阵流水线带宽、FP8 格式细节、稀疏 metadata 格式、dense/sparse 分拆 benchmark，以及任何 LLM/GEMM/卷积端到端数据。因此这份 PPT 能证明“RDNA 4 有明显加强的本地低精度矩阵路径”，不能精确证明某个模型或矩阵尺寸能达到多少 TFLOP/s。

## 8. 我的判断

- **对通用 GPU 计算理解：高。** 它清楚展示了 SIMD、scalar、VGPR、LDS、cache、矩阵单元如何在一个 CU 级执行块中配合。
- **对矩阵 kernel 优化价值：中到高。** 需要特别关注 `load and transpose → registers → WMMA → accumulator`，因为布局转换和寄存器压力常比峰值算力更早成为瓶颈。
- **对 AI 训练/服务价值：中。** FP8/INT4 和稀疏支持有价值，但 RDNA 4 的显存容量、互联和软件栈定位仍不同于 CDNA；PPT 没有 prefill、decode、KV cache、batching 或多 GPU 通信数据。
- **对图形系统内容：不在本文范围。** 光栅化、光线追踪、显示和媒体引擎均已省略。
- **是否值得复现：中。** 可用 HIP/ROCm 实现 elementwise、reduction 与 tiled GEMM，分别测量 wave 对齐、VGPR/LDS 占用、访问合并和 FP16/FP8/INT8 的带宽/误差；不应直接以厂商峰值替代 profiler 结果。

## 9. 下一步阅读建议

1. 回看原文第 5 页 Compute Engine 图，先建立 `wave32 → SIMD32 → CU → Dual CU/WGP` 的层次，再理解 VGPR、SGPR、LDS 分别存什么。
2. 精读第 11 页 Matrix Accelerator 图：重点是输入格式、寄存器中转和 accumulator，而不是图中的 sparse peak 数字。
3. 若要写 kernel，下一步读 ROCm/HIP 的 occupancy、LDS、wavefront 和 matrix API 文档，再用 profiler 测量 VGPR、LDS、cache miss、memory coalescing 与 matrix utilization。
4. 若目标是大模型训练/推理，接着对照 [AMD CDNA 4 白皮书笔记](amd-cdna-4-architecture-whitepaper.md)：它更适合讨论 HBM、矩阵吞吐、chiplet、Infinity Fabric 和多 GPU 系统。
