# CuTe 入门

CuTe 是一组 C++ CUDA 模板抽象，用于定义和操作线程与数据的层次化多维布局。CuTe 提供 `Layout` 和 `Tensor` 对象，用紧凑方式打包数据的类型、shape、内存空间和 layout，同时替用户处理复杂索引。这样程序员可以专注于算法的逻辑描述，而由 CuTe 完成机械性的簿记工作。借助这些工具，我们可以快速设计、实现和修改各种 dense linear algebra 操作。

CuTe 的核心抽象是层次化多维 layout。layout 可以与数据数组组合来表示 tensor。layout 的表达能力足以表示实现高效 dense linear algebra 所需的几乎所有结构。layout 也可以通过函数复合进行组合和变换，在此基础上构建出 tiling、partitioning 等大量常见操作。

## 系统要求

CuTe 共享 CUTLASS 3.x 的软件要求，包括 NVCC 以及支持 C++17 的 host compiler。

## 知识前提

CuTe 是一个 header-only CUDA C++ 库。它要求 C++17，也就是 2017 年发布的 C++ 标准修订版。

本教程假设读者具有中级 C++ 经验。例如，我们假设读者知道如何读写模板函数和模板类，以及如何使用 `auto` 关键字推导函数返回类型。我们会以温和方式介绍 C++，并解释一些你可能已经知道的内容。

我们也假设读者具有中级 CUDA 经验。例如，读者需要知道 device code 和 host code 的区别，以及如何 launch kernel。

## 构建测试和示例

CuTe 的测试和示例会作为 CUTLASS 正常构建流程的一部分进行构建和运行。

CuTe 的单元测试位于 [`test/unit/cute`](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute) 子目录。

CuTe 的示例位于 [`examples/cute`](https://github.com/NVIDIA/cutlass/tree/main/examples/cute) 子目录。

## 库组织

CuTe 是 header-only C++ 库，因此没有需要构建的源代码。库头文件位于顶层 [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) 目录中，组件按语义分组到不同目录。

| 目录 | 内容 |
|---|---|
| [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) | 顶层每个头文件对应 CuTe 的一个基础构件，例如 [`Layout`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp) 和 [`Tensor`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/tensor.hpp)。 |
| [`include/cute/container`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container) | 类 STL 对象的实现，例如 tuple、array 和 aligned array。 |
| [`include/cute/numeric`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric) | 基础数值数据类型，包括非标准浮点类型、非标准整数类型、复数和 integer sequence。 |
| [`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm) | copy、fill、clear 等工具算法的实现；如果可用，会自动利用架构特定特性。 |
| [`include/cute/arch`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch) | 架构特定 matrix-matrix multiply 和 copy 指令的 wrappers。 |
| [`include/cute/atom`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom) | `arch` 中指令的元信息，以及 partitioning、tiling 等工具。 |

## 教程

本目录包含 Markdown 格式的 CuTe 教程。[`0x_gemm_tutorial.md`](./0x_gemm_tutorial.md) 解释如何使用 CuTe 组件实现 dense matrix-matrix multiply。它给出 CuTe 的整体概览，因此适合作为起点。

本目录中的其他文件讨论 CuTe 的具体部分。

* [`01_layout.md`](./01_layout.md) 描述 `Layout`，也就是 CuTe 的核心抽象。

* [`02_layout_algebra.md`](./02_layout_algebra.md) 描述更高级的 `Layout` 操作和 CuTe layout algebra。

* [`03_tensor.md`](./03_tensor.md) 描述 `Tensor`，它是将 `Layout` 与数据数组组合起来的多维数组抽象。

* [`04_algorithms.md`](./04_algorithms.md) 总结 CuTe 中作用于 `Tensor` 的通用算法。

* [`0t_mma_atom.md`](./0t_mma_atom.md) 展示 CuTe 如何为 NVIDIA GPU 架构特定的 Matrix Multiply-Accumulate（MMA）指令提供元信息和接口。

* [`0x_gemm_tutorial.md`](./0x_gemm_tutorial.md) 从零开始演示如何用 CuTe 构建 GEMM。

* [`0y_predication.md`](./0y_predication.md) 解释当 tiling 不能整除矩阵时应该怎么处理。

* [`0z_tma_tensors.md`](./0z_tma_tensors.md) 解释 CuTe 用于支持 TMA loads 和 stores 的高级 `Tensor` 类型。

## 快速提示

### 如何在 host 或 device 上打印 CuTe 对象？

`cute::print` 函数对几乎所有 CuTe 类型都有 overload，包括 Pointers、Integers、Strides、Shapes、Layouts 和 Tensors。不确定时，可以先尝试对对象调用 `print`。

CuTe 的 print 函数可以在 host 或 device 上工作。注意，在 device 上打印代价很高。即使只是把 print 代码留在 device 代码中，哪怕它从未被调用，例如放在运行时不会进入的 `if` 分支里，也可能生成更慢的代码。因此，调试结束后务必移除 device 上的打印代码。

你也可能只想在每个 threadblock 的 thread 0 上打印，或者只在 grid 的 threadblock 0 上打印。`thread0()` 函数只在 kernel 的全局 thread 0 上返回 true，也就是 threadblock 0 的 thread 0。打印 CuTe 对象的常见写法是只在全局 thread 0 上打印。

```c++
if (thread0()) {
  print(some_cute_object);
}
```

有些算法依赖某个特定 thread 或 threadblock，因此你可能需要在非零 thread 或 threadblock 上打印。[`cute/util/debug.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/util/debug.hpp) 头文件提供多种工具，其中包括 `bool thread(int tid, int bid)` 函数：当当前执行位置是 thread `tid` 且 threadblock `bid` 时返回 `true`。

#### 其他输出格式

一些 CuTe 类型有特殊打印函数，会使用不同的输出格式。

`cute::print_layout` 函数会用纯文本表格显示任意 rank-2 layout。这很适合可视化从坐标到 index 的映射。

`cute::print_tensor` 函数会用纯文本多维表格显示任意 rank-1、rank-2、rank-3 或 rank-4 tensor。tensor 的值也会被打印出来，因此你可以验证例如 copy 后得到的数据 tile 是否符合预期。

`cute::print_latex` 函数会打印 LaTeX 命令，你可以用 `pdflatex` 构建格式美观、带颜色的表格。它适用于 `Layout`、`TiledCopy` 和 `TiledMMA`，对于理解 CuTe 内部的 layout pattern 和 partitioning pattern 很有帮助。

## Copyright

Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```text
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
