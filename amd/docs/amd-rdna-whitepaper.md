# AMD RDNA Architecture 白皮书精读

> 原文：[amd-rdna-whitepaper.pdf](../pdfs/amd-rdna-whitepaper.pdf)  
> 文档性质：2019 年 AMD 面向首代 RDNA（Navi / Radeon RX 5700 系列）的架构白皮书，不是同行评审论文。性能数字主要是峰值吞吐或厂商代际对比，不能直接当作游戏帧率或某个 kernel 的实测加速比。

## 论文一句话总结

RDNA 的核心不是单纯增加 ALU，而是把 GCN 的 **wave64 + 单 CU 为中心** 改造成 **原生 wave32 + 两个相邻 CU 协作的 Dual Compute Unit**，并在 L0 和全局 L2 之间增加共享 Graphics L1。它要解决的是游戏 shader 控制流更复杂、缓存和寄存器供给跟不上计算的问题。

从实现视角看，RDNA 让一个 32 线程 wave 可以每周期在 32-lane SIMD 上完成一条向量指令，同时把 L0 miss 和像素管线请求先汇聚到 shader array 内的 L1，减少昂贵的 L2/显存访问；代价是软件和编译器需更好地利用 wave32、局部性和 LDS，且白皮书没有给出真实应用中各瓶颈的独立贡献。

## 1. 背景和问题

GCN 以 64 work-item 的 wave64 为基本执行粒度，适合规则、数据并行度高的计算，但现代游戏 shader 有更多分支、循环和计算式效果。分支一旦让 wave 内线程走不同路径，未参与当前路径的 lane 只能闲置；同时，更宽的执行单元也需要更高的寄存器、缓存和访存带宽。RDNA 的目标是在保留 GCN 指令类型和 wave64 兼容性的前提下，提高复杂 shader 的有效利用率、降低数据移动，并让图形与异步计算能更可靠地共存。

## 2. 核心结论

- 原生执行粒度从 GCN 的 wave64 扩展为以 **wave32 为主**；wave64 仍支持，但由两个 wave32 半波执行。更小的 wave 降低分歧浪费、寄存器占用与完成延迟。
- 基本计算积木从一个 CU 转为 **Dual Compute Unit**：4 个独立 SIMD，每个 SIMD 32 个 vector ALU；一个 Dual CU 合计每周期可持续 256 个 FP32 FLOP，外加 scalar 运算。
- SIMD 不再像旧 GCN 那样以四周期轮转发射；每个 SIMD 可每周期 decode/issue。对可并行的指令流，这减少了 issue 延迟并提高 ILP 利用。
- 数据通路变为 **SIMD 私有/近端 L0 → shader-array 共享 Graphics L1 → 全局分片 L2 → GDDR6**。新增的 128 KB Graphics L1 是架构平衡的关键：它把一组 Dual CU 与 pixel pipeline 的 miss 聚合，降低 L2 压力和功耗。
- RX 5700 XT 以 40 CU、9.75 TFLOP/s FP32 对比 Vega 64 的 64 CU、12.66 TFLOP/s，原始 FP32 峰值反而较低；但总 cache 带宽为 15.61 TB/s，对应 1.6 bytes/FLOP，约为 Vega 64 的 2.5 倍。这说明设计重点是让较少的算力更少挨饿。
- AMD 的代际整机口径是：相同功耗下相对 14 nm Vega 64 可提升约 50% 性能，RX 5700 XT 对比 Vega 64 有 1.5 倍性能/瓦和 2.3 倍性能/面积。该结果同时包含 7 nm 工艺、频率和微架构变化，不能归因给某一项 RDNA 技术。

## 3. 形象解释 / Mental Model

### 3.1 从“64 人大队”到“32 人小队”

把一个 wavefront 想成一组必须同步行动的工人。GCN 的 wave64 是 64 人大队：如果一半人遇到 `if` 分支，先走 A 路的 32 人工作时，另外 32 人只能等待；随后再反过来。RDNA 的 wave32 是 32 人小队，同样的空间区域被拆成更多小队，分支更可能只影响一小队，寄存器也会更早释放给下一批工作。

这不是说 wave32 消除了 SIMT 分歧。它仍以 mask 屏蔽不活跃 lane，仍要串行执行不同分支；它减少的是每次被迫闲置的 lane 数量和每个 wave 占用的状态量。真实硬件对象是：一个 work-group 被编译器划分为 wave32 或 wave64；每个 wave 有 program counter、执行掩码、scalar/vector register 分配和 cache 请求状态。

### 3.2 Dual CU 像四条独立的 32 车道生产线

RDNA 把两个相邻 CU 作为一个协作块，但并没有把它们变成一条锁步的大流水线。一个 Dual CU 内有 4 个互相独立的 SIMD32；每条 SIMD 配有自己的 wavefront controller、scalar pipeline 和 vector ALU，附近共享指令/数据通路的一部分资源。

可以记住下面这张“图”：

```text
一个 Dual Compute Unit
  ├─ SIMD0: wave controller + scalar + 32-lane vector ALU
  ├─ SIMD1: wave controller + scalar + 32-lane vector ALU
  ├─ SIMD2: wave controller + scalar + 32-lane vector ALU
  └─ SIMD3: wave controller + scalar + 32-lane vector ALU
       ↓ 读写
  L0 instruction / scalar cache / 两个 L0 vector cache / LDS
       ↓
  Shader Array 的 shared Graphics L1
       ↓
  全局分片 L2 → memory controller → GDDR6
```

### 3.3 一次像素 shader wave 如何流动

以一个 wave32 的 pixel shader 为例：

1. 命令处理器把 work-group 分派到一个 shader array，硬件将其拆成 wave32；该 wave 被放到某个 SIMD 的 20-entry wavefront controller 中。
2. SIMD 从共享的 32 KB L0 instruction cache 取指。一个 SIMD 每周期可请求，I-cache 每周期可向每个 SIMD 提供 32 B 指令。
3. 每个 lane 从 VGPR 读取颜色、坐标等不同数据；全 wave 共享的地址计算、分支条件等尽量走 scalar ALU/SGPR，避免 32 个 lane 重复做同一件事。
4. texture load 先访问该 Dual CU 的 16 KB L0 vector cache；miss 会到该 shader array 的 128 KB Graphics L1，再 miss 才进入全局 L2 和 GDDR6。
5. 32-lane vector ALU 在一个周期完成该 wave 的普通 FP32 FMA。若部分像素被 branch mask 关掉，只有有效 lane 写回结果。
6. 最终颜色经 export 发给 Render Backend，完成深度/模板/混合；或写入全局数据存储 GDS。

真正移动的瓶颈是两处：**控制流粒度**从 64 降到 32，和 **供数路径**从“CU L0 直接打到全局 L2”变成“先在 shader array 内复用的 L1 汇聚”。

## 4. 方法总览

### 4.1 Baseline：GCN wave64 与四个 SIMD16

GCN 的一个 CU 有 4 个 16-lane SIMD。一个 wave64 向量指令需要分四个 cycle 执行完，且旧前端采用轮转策略，一个 SIMD 的 issue 机会隔四个 cycle 才来一次。为把 ALU 填满，硬件需要维持足够多 active wave 来隐藏 ALU、memory 和控制流延迟。

RDNA 保留 scalar compute、scalar memory、vector compute、vector memory、branch、export、message 七类 GCN 指令，但重排执行和存储层次：

| 维度 | GCN | RDNA 首代 |
|---|---|---|
| 主执行粒度 | wave64 | wave32 原生，兼容 wave64 |
| SIMD 宽度 | 16 lanes | 32 lanes |
| 基本组织 | 单 CU | 两个相邻 CU 组成 Dual CU |
| 单 wave 指令完成 | wave64 需 4 cycle | wave32 通常 1 cycle；wave64 由两半完成 |
| 指令 issue | SIMD 轮转，约每 4 cycle 一次 | SIMD 可每 cycle decode/issue |
| cache 路径 | CU 私有 L0 → 全局 L2 | L0 → array 共享 Graphics L1 → 全局 L2 |

### 4.2 核心吞吐关系

对每个 SIMD 的普通 FP32 FMA，32 个 lane 各做一次乘加，FMA 按两个 FLOP 计：

$$
\text{FP32 FLOP/cycle per SIMD} = 32 \text{ lanes} \times 2 = 64
$$

一个 Dual CU 有 4 个 SIMD，因此：

$$
\text{FP32 FLOP/cycle per Dual CU} = 4 \times 32 \times 2 = 256
$$

这里的 256 是峰值算术吞吐，成立前提是连续的 FMA 指令、足够 occupancy、没有 register/LDS 冲突，且 L0/L1/L2/显存能及时供数。实际 shader 往往被 texture、分支、导出、寄存器压力或 ROP 限制。

### 4.3 Cache 与内存层次

| 层级 | 范围与规格 | 实际作用 |
|---|---|---|
| L0 instruction | 每 Dual CU 32 KB，4-way、4 bank、64 B line | 供 4 个 SIMD 取指，降低复杂 shader 的取指竞争 |
| scalar cache | 每 Dual CU 16 KB，4-way、2 bank、64 B line | 服务 wave 内共享的 scalar 地址、常量与控制数据 |
| L0 vector cache | 每 Dual CU 两个各 16 KB，4-way、128 B line、write-through | 为成对 SIMD 提供 vector load / texture 数据；同一 work-group 内保持一致 |
| LDS | 两个 64 KB、各 32 bank SRAM | 显式寻址的 work-group scratchpad，承担 tile 复用、同步和原子操作 |
| Graphics L1 | 每 shader array 128 KB，16-way、4 bank、128 B line | 聚合 L0 miss 与 pixel engine 请求，服务最多 4 个请求/cycle |
| L2 | 靠近 memory controller 的全局分片 cache | 为整个 GPU 提供共享缓存；RX 5700 XT 总共 4 MB |
| GDDR6 | RX 5700 XT：256-bit、448 GB/s | 容量和最终带宽来源，访问延迟和能耗最高 |

对 kernel 作者而言，LDS 相当于 CUDA shared memory：可以明确控制 tile 和同步；L0/L1/L2 则依赖连续访问、数据复用和访问时间局部性。RDNA 的新增 Graphics L1 特别有利于临近 Dual CU/Render Backend 对相同数据的重复读取，但它是只读 cache，写入会使对应行失效并送往 L2/内存。

### 4.4 运行系统与图形管线

RX 5700 XT 用 Infinity Fabric 连接两个 Shader Engine；每个 Shader Engine 有两个 Shader Array。每个 Shader Array 包含一个 primitive unit、一个 Graphics L1、四个 render backend 和五个 Dual CU，整卡因此有 4 个 shader array、20 个 Dual CU、40 个 CU。

固定功能单元仍然关键：primitive unit 的 cull rate 提升到 2 primitive/clock；每个 rasterizer 每 clock 可处理一个 triangle、发出最多 16 个 pixel；每个 RB 每 clock 可测试、采样、混合 4 个输出 pixel。RDNA 不是“所有图形工作都变成 compute”，而是以更强的 compute shader 为中心，配合这些图形固定功能单元。

## 5. 关键算法 / 执行机制

### 5.1 wave32 分歧与执行掩码

设 wave 的活动 lane 掩码为 $M \in \{0,1\}^{W}$，其中 $W \in \{32,64\}$。一条向量指令在逻辑上可写为：

$$
y_i \leftarrow f(x_i), \qquad \text{only if } M_i = 1, \quad i \in [0, W)
$$

**作用。** 分支后的两个路径通过改变 $M$ 串行执行；这保证 SIMT 的语义正确，但非活动 lane 不产生有用工作。

**符号与 shape。** $x,y \in \mathbb{R}^{W}$ 是一个向量寄存器在一个 wave 中的 lane 值；$M$ 是长度为 $W$ 的 execution mask。RDNA 每个 VGPR 物理上恰对应 32 个 32-bit lane，所以 wave32 最自然；wave64 需要处理两个 32-lane half。

**直觉。** 掩码像一排总开关：同一条指令仍广播到所有 lane，但关掉开关的 lane 不写结果。把 $W$ 从 64 变为 32，不能避免绕路，却缩小了每次绕路时可能被关掉的工作集合。

**极小例子。** 若 64 个像素中 32 个满足 `roughness > 0.5`，wave64 要先以 mask 执行粗糙表面路径、再执行另一条路径。若像素被拆成两个 wave32 且数据局部性好，可能刚好一队大多走真分支，另一队大多走假分支，闲置更少；最坏情况下两个 wave 都各一半分歧，收益会变小。

**PyTorch 语义。** 这只是语义类比，不代表 GPU 实际以 PyTorch 实现：

```python
# x: [32], active: [32] bool
candidate = f(x)
y = torch.where(active, candidate, y)
```

**影响。** 对图形/通用 compute 都降低复杂控制流的浪费和 wave 的寄存器状态；它与 LLM decode 的单 token 执行不是直接对应关系，但同样体现了“小粒度并行单元更容易避免无效 lane”的原则。

### 5.2 scalar + vector 双路径

shader 通常混合 wave 内共享的工作和每个 lane 不同的工作。RDNA 将它们分流：

$$
a \leftarrow g(a) \quad \text{(scalar, one value per wave)}
$$

$$
\mathbf{v}' \leftarrow \operatorname{FMA}(\mathbf{v}, a, \mathbf{b}) \quad \text{(vector, } \mathbf{v},\mathbf{b}\in\mathbb{R}^{32}\text{)}
$$

**作用。** 把同一 wave 32 个 lane 都相同的地址计算、循环计数、分支目标等放在 scalar ALU/SGPR 中做一次，再将 $a$ 广播给 vector lane；每个 pixel/thread 的颜色、坐标和中间结果保留在 VGPR。

**shape。** 每 SIMD 有 1,024 个 32-lane VGPR；scalar register file 为每个 active wave 提供 128 个 32-bit scalar entry。每 SIMD 最多跟踪 20 个 wave，Dual CU 合计最多 80 个 wave 和 32 个 work-group。

**直觉。** Scalar 路径是“组长只算一次并通知全组”，vector 路径才是“32 个人各自处理不同数据”。如果编译器无法识别 uniform 值，原本只需一次的工作会被复制 32 次。

### 5.3 LDS 分 bank 的并行访问

一个 LDS 可抽象为 32 个 bank，每 bank 有独立的一读一写端口：

$$
\operatorname{bank}(a) = \left\lfloor \frac{a}{4} \right\rfloor \bmod 32
$$

这里 $a$ 是按 byte 计的 32-bit 对齐地址；实际映射还受数据类型、指令和布局影响，以上是理解 bank conflict 的简化模型。

**作用。** LDS 让 work-group 显式缓存可复用 tile、做 barrier 和 atomic；不同 lane 若落在不同 bank，可并行完成。多个 lane 请求同一 bank 的不同地址时会序列化；读取同一地址则可以 broadcast。

**极小例子。** 32 个 lane 读取 `lds[lane_id]` 时，连续的 4-byte 元素分别落入 32 个 bank，理想并行；读取 `lds[lane_id * 32]` 时，它们会反复映射到同一 bank，产生严重冲突。RDNA 的 LDS 还在每 bank 配有 ALU，用于部分 integer 和 floating-point atomic。

**影响。** GEMM、FFT、物理模拟或 compute-based effect 的性能常由 LDS tile 布局、同步次数和 VGPR/LDS 资源占用共同决定。LDS 并非自动缓存，代码没有把可复用数据搬进去，就不会获得它的带宽优势。

## 6. 数据流和实现设计

### 6.1 取指、调度与寄存器生命周期

每个 SIMD 有独立 instruction pointer 和 20-entry wavefront controller。RDNA 的 scheduler 使用 oldest work-group 优先策略，使较早 work-group 的 wave 更稳定地前进；这有助于它们在完成前保持局部 cache 数据，也避免某些 work-group 长时间得不到推进。

wave32 比 wave64 使用约一半 register state，并更快完成。因此在相同物理寄存器容量下，GPU 可能保持更多活跃 wave 来隐藏 memory latency。但这不是无条件收益：如果每个 thread 使用过多 VGPR 或每个 work-group 占用大量 LDS，occupancy 仍然下降，wave32 也无法凭空提供更多并发。

RDNA 还支持 clause，即软件把一段无显式 break 的指令标记为不可中断序列。load clause 的意图是让一批 vector load 使用完某条 cache line 后再被换出；compute clause 则帮助计算/写回与消费者依赖安排。这是 ISA/编译器侧配合 cache 局部性的机制，而不是用户 shader 中通常直接操作的高级特性。

### 6.2 双 L0 vector cache 与 texture 路径

一个 Dual CU 的两条 128-byte 数据总线各连接一对 SIMD、一个 16 KB L0 vector cache 和 texture filtering unit。两条总线聚合带宽是上代的四倍；单 SIMD 可以在同一时钟获得两条 128 B line，分别来自 LDS 和 L0。

buffer/image load 会先做地址合并：同一个 128 B line 内的 lane 请求尽量合为一次 transaction。没有采样需求的 image load 可以绕过 texture interpolation，以 buffer load 的速率执行；白皮书称其相对旧路径的吞吐提高 8 倍、L0 hit latency 降低 35%。这提示 kernel 作者：若只需读取 texture/image 的原始数据，不要无谓使用采样路径。

纹理单元最多每 clock 过滤 8 个 texture address，64-bit 双线性过滤吞吐翻倍。它还能将增强的 delta color compression 数据以压缩形式写回 L2/后续层级，减少颜色缓冲的带宽占用。

### 6.3 Graphics L1 到 L2 的汇聚

Graphics L1 是每个 shader array 一份的只读 128 KB cache，位于多个 Dual CU 和 render backend 的共同上游。其工作流为：

```text
SIMD texture/vector load 或 RB pixel request
  → 对应 Dual CU 的 L0（若有）
  → Shader Array Graphics L1（128 KB）
  → partitioned L2
  → memory controller / GDDR6
```

这样做的收益并不只是“多了一层 cache”：以前许多局部 miss 都会集中轰击全局 L2；现在同一 shader array 内的数据可以在 L1 复用，L2 的交叉互联与远距离搬运也减少。代价是 L1 容量有限、只读且写入导致 invalidation；两个 L0 vector cache 之间的 coherency 也由软件负责，跨越它们共享数据时要用正确同步/一致性策略。

### 6.4 图形、异步计算和 QoS

graphics command processor 管传统图形管线，Asynchronous Compute Engine（ACE）管理独立 compute queue。普通情况下，graphics shader 和 compute shader 可在 CU 间交织推进。RDNA 的 Asynchronous Compute Tunneling 更进一步：当某个任务极端看重 latency 时，硬件可以完整 suspend 其他 shader，把全部 CU 腾给高优先级任务。

它适合 VR 或 TrueAudio Next 这类 deadline 明确的任务。代价同样明确：被暂停的任务吞吐下降，状态保存/恢复和调度策略需要驱动保证正确性。因此它是 QoS/尾延迟工具，不是让总吞吐无成本增加的机制。

## 7. 实验和效果

白皮书没有标准论文式 benchmark suite，而是以 RX 5700 XT 与 Vega 64 的峰值资源、图形速率和整卡指标说明设计意图。

| 指标 | RX 5700 XT（RDNA） | RX Vega 64（GCN） | 应如何理解 |
|---|---:|---:|---|
| CU | 40 | 64 | RDNA 不是依靠更多 CU |
| FP32 peak | 9.75 TFLOP/s | 12.66 TFLOP/s | 原始 FP32 峰值更低，不代表实际 shader 更慢 |
| 总 cache bandwidth | 15.61 TB/s | 7.91 TB/s | RDNA 的计算供数显著更强 |
| cache bandwidth / FLOP | 1.6 B/FLOP | 0.625 B/FLOP | 每单位峰值算力对应约 2.56 倍 cache 带宽 |
| 纹理 FP16 | 304.8 GTexel/s | 197.9 GTexel/s | HDR/FP16 采样不再显著吃亏 |
| 三角形 cull | 15,240 Mtri/s | 6,184 Mtri/s | 减少不可见几何进入后续管线 |
| 显存带宽 | 448 GB/s | 484 GB/s | 外部带宽略低，依赖更好的 on-chip cache |
| die area | 251 mm² | 495 mm² | 同时受工艺节点、产品定位影响 |
| TDP | 225 W | 295 W | 整卡能效对比，不是纯微架构指标 |

最有说服力的不是 1.5x 性能/瓦宣传数字，而是“少 24 个 CU、较低 FP32 峰值，却配置约两倍总 cache 带宽”的资源重配。这支持 RDNA 的核心论点：复杂 shader 往往受访存、控制和 pipeline 平衡限制，不能仅以 TFLOP/s 判断。

限制也很明显：没有给出相同频率/工艺的隔离实验，没有 wave32 与 Graphics L1 的单独消融，没有具体游戏场景的 cache hit rate、branch efficiency、occupancy、frame-time 分位数，也没有公开 kernel 指令时序。因此不应把表中峰值转换为任何特定程序的确定加速倍数。

## 8. 我的判断

- **真正贡献：高。** RDNA 的重要性在于重新定义了 AMD 游戏 GPU 的执行粒度和缓存边界：wave32、每周期 SIMD issue、Dual CU、array 共享 L1 共同构成一个平衡设计，而非孤立特性。
- **最值得精读：高。** 第 4–5 页的 wave32/Dual CU 动机，第 9–17 页的前端、SIMD、LDS 和 vector cache，以及第 17–18 页的 Graphics L1/L2。它们能建立“shader 如何被调度、如何被喂饱”的完整模型。
- **可快速浏览：中。** 视频、显示、TrueAudio Next 和 FidelityFX CAS 章节有产品价值，但对理解 RDNA 执行核心不是必要条件。
- **对算法理解价值：低到中。** 它不提出新的 AI/图形算法；其价值是解释算法在 GPU 上为何会因为分歧、数据局部性和精度路径而表现不同。
- **对推理系统价值：中。** 其中没有 KV cache、continuous batching 等服务系统内容，但“decode 常受带宽约束”“提升算力需配套 cache/LDS/调度”的硬件直觉可迁移到推理分析。
- **对 kernel / 算子优化价值：高。** wave32 选择、uniform scalar 化、128 B 合并访问、LDS bank conflict、L0/L1 数据复用和 VGPR/LDS 对 occupancy 的约束，都直接影响 HIP/GCN ISA kernel 性能。
- **是否值得复现：低。** 这是硬件架构而非可独立复现的算法。更实际的练习是用 HIP 写两个 wave32 kernel，对比连续/跨步访问、是否使用 LDS，以及分支重排前后的性能与 profiler 指标。

## 9. 下一步阅读建议

1. 先回看原文 Figure 2–3：确认“两个 CU 组成 Dual CU，但四个 SIMD 独立”的含义，并把 wave32/wave64 的 issue 时序与 GCN 的 SIMD16 对比起来。
2. 精读 Figure 10 和 LDS 章节：画出 `SIMD32 → LDS/Tex unit → Vector L0 → L1/L2` 数据路径，再用一个 $32 \times 32$ tiled GEMM 思考哪些数据该驻留 LDS、哪些留在 VGPR。
3. 若目标是写 RDNA kernel，下一步应读 AMDGPU ISA 和 ROCm/HIP 的 wavefront、LDS、occupancy 文档，并用 profiler 验证 VGPR 数、LDS 分配、cache hit 与 memory coalescing，而非仅看 TFLOP/s。
4. 若目标是理解后续 AMD GPU 演进，可接着读现有的 [AMD CDNA 4 白皮书笔记](amd-cdna-4-architecture-whitepaper.md)：它将同一类 CU/LDS/缓存思想转向矩阵计算、HBM 和多 GPU AI 系统。
