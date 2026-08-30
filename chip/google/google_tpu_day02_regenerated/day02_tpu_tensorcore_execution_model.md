# Google TPU Day 2：TensorCore 执行模型——MXU、VPU、Scalar Unit 到底怎么协作

- 日期：2026-08-27
- 预计学习时间：约 30 分钟
- 承接 Day 1：昨天把 `chiplet → chip → device → VM → slice → Pod` 放对层级；今天只往一个 compute chiplet 里面钻一层。
- 今日边界：只讲 TensorCore 的执行模型和 MXU/VPU/Scalar Unit 分工。VMEM/HBM 的完整层级、DMA pipeline、SparseCore 都留到后续 Day。

## 今日目标

学完只要求形成三个判断：

1. Google 的 `TensorCore` 为什么不能类比成 NVIDIA 的一个 Tensor Core；
2. 一个算子里的 matmul、elementwise/reduction、scalar/control 分别落到 MXU、VPU、Scalar Unit 的哪一侧；
3. 为什么 TPU 对程序员看起来近似“单线程”，硬件内部却可以让多个执行单元异步推进。

## 1. 先看最有用的一张 TensorCore 图

![JAX Scaling Book：TPU TensorCore 抽象图](https://jax-ml.github.io/scaling-book/assets/img/tpu.png)

来源：[JAX Scaling Book — How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/)

**怎么看：**

- 左边整个灰框才接近一个 TPU `TensorCore`；其中包含 Scalar Unit、VPU、MXU 和 VMEM。
- HBM 在 TensorCore 外；Pallas 的典型计算路径不是从 HBM 直接算，而是先把工作 tile 搬进 VMEM。
- MXU 专门吞矩阵块，VPU 负责 elementwise / reduction 等向量工作。
- 图是抽象执行图，不是 Ironwood floorplan。

JAX 当前官方 Pallas 文档把 TensorCore 分成三类对象：memory spaces、registers、compute units。TensorCore 有 VREG/SREG 两类寄存器；compute units 包括 Scalar Unit、VPU、MXU。这些单元可以异步工作，但由 TPU compiler 管理，所以程序员视角仍近似单线程。

## 2. `TensorCore` 的正确粒度：它是一个 compute subsystem

Google Cloud 对 TPU chip 的统一描述可以压成：

```text
TPU chip
└── one or more TensorCores
    ├── one or more MXUs
    ├── VPU
    └── Scalar Unit
```

Ironwood 一颗物理 chip 有两个 TensorCore；dual-chiplet 设计中，每个 compute chiplet 对应一个 TensorCore，并作为一个 framework-visible device 暴露。

最容易犯的错是名字类比：

```text
Google TensorCore != NVIDIA Tensor Core
```

NVIDIA Tensor Core 是 SM 内的矩阵 MMA execution unit；Google TensorCore 则包含矩阵、向量、标量、寄存器和本地 SRAM 等一整套执行资源。

| Google TPU | NVIDIA 对照 | 注意 |
|---|---|---|
| TensorCore | 较大的 compute subsystem | 不是严格对应 SM |
| MXU | Tensor Core / MMA pipe | 都负责 dense matrix math |
| VPU | SM 内 vector/ALU execution pipes | TPU 分工更显式 |
| Scalar Unit | scalar/control 侧资源 | 不机械映射某一个 NVIDIA 单元 |
| VMEM | programmer-visible on-chip SRAM | 更接近显式 scratchpad，不是普通透明 cache |

## 3. MXU：TPU 的 FLOPS 主要从哪里来

Google 当前 Cloud TPU 架构文档给出：TPU v6e 和 TPU7x 的 MXU 使用 `256 × 256` systolic array；更早版本主要是 `128 × 128`。MXU 提供 TensorCore 的主要计算能力。

### 3.1 systolic array 先只理解一个工程点

![Google 官方：第一代 TPU systolic array](https://storage.googleapis.com/gweb-cloudblog-publish/images/tpu-17u39j.max-500x500.PNG)

来源：[Google Cloud — An in-depth look at Google’s first TPU](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)

**怎么看：**

- 数据进入阵列后规律传播并复用，MAC 阵列在数据流过时持续累加；目标是降低反复访问大存储的代价。
- 这解释了为什么 TPU 喜欢规则的大矩阵块，也解释了 XLA/Pallas 为什么非常关心 tiling。
- 不要把第一代图里的具体尺寸套到 Ironwood；TPU7x 当前公开的是 256×256 MXU。

对 kernel 工程而言，第一层问题不是“开多少 thread”，而是：

```text
什么 tile 放进 VMEM？
        ↓
怎样形成 MXU 友好的矩阵块？
        ↓
MXU 工作时 VPU / DMA 能否推进别的阶段？
```

## 4. VPU：matmul 快不代表整个 Transformer 就快

Transformer 里还有大量 activation、add/mul、normalization、reduction、packing、部分 quant/dequant。Google 的 Ironwood 软件栈资料把 MXU 定位为 matrix operations 的核心，把 VPU 定位为 element-wise operations，例如 activation 和 normalization。

可能出现：

```text
MXU: GEMM
VPU: post-op
MXU: 等 VPU / data
```

也可以通过流水让不同资源重叠：

```text
MXU: GEMM tile0 | GEMM tile1 | GEMM tile2
VPU:            | post tile0 | post tile1
DMA: load tile1 | load tile2 | load tile3
```

JAX Pallas 官方文档明确说明 MXU、VPU、Scalar Unit 可以异步运行，compiler 负责管理。

## 5. Scalar Unit：程序“单线程”不等于硬件串行

CUDA kernel 首先暴露：

```text
grid → CTA → warp → lane/thread
```

Pallas TPU 更接近：

```text
一个 program 的顺序语义
        ↓ compiler lowering / scheduling
Scalar / VPU / MXU / DMA
        ↓
硬件异步重叠
```

Pallas 还规定：0D array 放 scalar registers，并在 scalar core 执行；1D+ array operation 则在 vector core 执行。

所以“single-threaded”描述的是编程语义，不是芯片一次只能做一件事。

## 6. 寄存器与 VMEM：今天只建立数据路径

JAX Pallas 的公开模型可以记成：

```text
HBM
 ↓ copy / DMA
VMEM → VREG → VPU / MXU
              ↓
             VREG
              ↓
             VMEM
              ↓
             HBM

SMEM → SREG → Scalar Unit
```

其中 `VMEM` 是 vector SRAM，`SMEM` 是 scalar SRAM，`VREG` / `SREG` 是对应寄存器。

JAX Hardware Reference 当前按**每 TensorCore**列出 Ironwood 约 `64 MiB VMEM`、`1 MiB SMEM`。一颗 Ironwood chip 有两个 TensorCore，因此不能把 per-TensorCore 数字直接当 per-chip。

## 7. NVIDIA 对照：差别主要在“机器模型怎么暴露”

NVIDIA Hopper/Blackwell 同样依赖 specialized matrix execution。NVIDIA kernel programmer 会直接面对 CTA、warpgroup、warp、register、shared memory、TMA、MMA instruction、barrier/pipeline。

TPU 更强调 program、memory space、tile、MXU/VPU、DMA 和 compiler-managed scheduling。

两边都要解决 tile、pipeline、operand reuse；区别在于**程序员和 compiler 各承担多少调度细节**。

## 8. AI workload 映射：RMSNorm + GEMM

考虑：

```text
x → RMSNorm → Linear/GEMM → activation
```

| 阶段 | TPU 主要资源 |
|---|---|
| RMS reduction / elementwise | VPU |
| scalar loop/index/control | Scalar Unit |
| weight × activation | MXU |
| tile staging | VMEM |
| HBM ↔ VMEM | DMA/data movement |

如果阶段没有重叠好，即使 MXU 峰值很高，也可能被 VPU 或数据供应制造 bubble。

## 9. 容易混淆的四点

1. `MXU = TensorCore`？不是，MXU 是 TensorCore 内的 matrix unit。
2. `VPU = CUDA Core`？只能功能级粗类比，不能做数量/调度层硬映射。
3. `TPU program single-threaded = 硬件串行`？不是，多个 unit 可以异步推进。
4. `VMEM = GPU L1 cache`？不准确。VMEM 是显式暴露、可由 Pallas 管理的 on-chip vector SRAM/scratchpad。

## 结论

今天最重要的一句话：

> Google TensorCore 是由 MXU、VPU、Scalar Unit、寄存器和本地 memory space 组成的 specialized compute subsystem；TPU 的性能来自 compiler 把矩阵、向量、标量和数据搬运分派并重叠，而不是依赖大量 CUDA-style threads 自己争取执行槽。

## 验收标准

1. 能解释 Google `TensorCore → MXU/VPU/Scalar Unit` 与 NVIDIA `SM → Tensor Core/other pipes` 为什么不能一一对应。
2. 能口述 `HBM → VMEM → VREG → MXU/VPU` 与 `SMEM → SREG → Scalar Unit` 两条基本数据路径。
3. 能解释“TPU 程序员视角单线程，但硬件多个 unit 可异步”的含义。

**下一课：Google TPU Day 3 —— VMEM / SMEM / HBM：Pallas memory spaces、DMA 与 double buffering，重点解释为什么 TPU kernel 首先是数据搬运问题。**

## 参考资料

1. [JAX Pallas — TPU Pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
2. [JAX Pallas — TPU Hardware Reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)
3. [Google Cloud — TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
4. [Google Cloud — TPU7x (Ironwood)](https://docs.cloud.google.com/tpu/docs/tpu7x)
5. [Google Cloud Blog — Inside the Ironwood TPU codesigned AI stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack/)
6. [JAX Scaling Book — How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/)
7. [NVIDIA — Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
