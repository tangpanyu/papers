# CuTe Tensor 算法

本节总结作用于 `Tensor` 的常见数值算法的接口和实现。

这些算法的实现位于 [include/cute/algorithm/](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/) 目录。

## `copy`

CuTe 的 `copy` 算法会把 source `Tensor` 的元素复制到 destination `Tensor` 的元素中。`copy` 的各种 overload 位于 [`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp)。

### 接口和特化机会

`Tensor` 会在编译期封装 tensor 的数据类型、数据位置，以及可能的 shape 和 stride。因此，`copy` 可以并且确实会根据参数类型 dispatch 到各种同步或异步硬件 copy 指令。

`copy` 算法有两个主要 overload。第一个只接受 source `Tensor` 和 destination `Tensor`。

```c++
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

第二个接受这两个参数，外加一个 `Copy_Atom`。

```c++
template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

两参数版本的 `copy` 只根据两个 `Tensor` 参数的类型选择默认实现。`Copy_Atom` overload 允许调用者指定非默认 `copy` 实现，从而覆盖默认选择。

### 并行性和同步取决于参数类型

默认实现或 `Copy_Atom` overload 选择的实现可能完全不使用并行性，也可能使用所有可用并行性；它也可能具有不同的同步语义。行为取决于 `copy` 的参数类型。用户需要根据自己对目标架构的了解判断这些行为。开发者通常会为每种 GPU 架构写自定义优化 kernel。

`copy` 算法可能对每个线程是顺序执行的，也可能跨某个线程集合并行执行，例如 block 或 cluster。

如果 `copy` 是并行的，那么参与复制的线程集合可能需要同步，然后其中任意线程才能假定 copy 操作已经完成。例如，如果参与线程组成一个 thread block，那么用户必须调用 `__syncthreads()` 或 Cooperative Groups 中的等价操作，然后才能使用 `copy` 的结果。

`copy` 算法可能使用异步 copy 指令，例如 `cp.async`，或其 C++ 接口 `memcpy_async`。这种情况下，用户需要在使用 `copy` 结果前执行该底层实现所需的额外同步。[CuTe GEMM tutorial example](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/) 展示了一种这样的同步方法。更优化的 GEMM 实现会使用流水线技术，将异步 `copy` 操作与其他有用工作重叠。

### 一个通用 copy 实现

任意两个 `Tensor` 的简单通用 `copy` 实现示例如下。

```c++
template <class TA, class ALayout,
          class TB, class BLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<TA, ALayout> const& src,  // Any logical shape
     Tensor<TB, BLayout>      & dst)  // Any logical shape
{
  for (int i = 0; i < size(dst); ++i) {
    dst(i) = src(i);
  }
}
```

这个通用 `copy` 算法用一维逻辑坐标访问两个 `Tensor`，因此会以逻辑 column-major 顺序遍历两个 `Tensor`。一些合理的架构无关优化包括：

1. 如果两个 `Tensor` 具有已知 memory space，并且该 memory space 有优化访问指令，例如 `cp.async`，则 dispatch 到自定义指令。

2. 如果两个 `Tensor` 具有 static layout，并且可以证明元素向量化是合法的，例如四个 `ld.global.b32` 可以组合成单个 `ld.global.b128`，则对 source 和 destination tensor 向量化。

3. 如果可能，验证即将使用的 copy 指令是否适合 source 和 destination tensor。

CuTe 的优化 copy 实现可以完成所有这些事情。

## `copy_if`

CuTe 的 `copy_if` 算法和 `copy` 位于同一个头文件 [`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp)。该算法像 `copy` 一样接受 source 和 destination `Tensor` 参数，但还接受一个与输入输出 shape 相同的 “predication `Tensor`”。只有当对应的 predication `Tensor` 元素非零时，source `Tensor` 的元素才会被复制。

关于为什么以及如何使用 `copy_if`，请参考教程中的 [“predication” 小节](./0y_predication.md)。

## `gemm`

### `gemm` 计算什么

`gemm` 算法接受三个 `Tensor`：A、B 和 C。它做什么取决于这些 `Tensor` 参数有多少个 modes。我们用字母表示这些 modes。

* V 表示 “vector”，也就是一组独立元素的 mode。

* M 和 N 分别表示 BLAS GEMM 例程中矩阵结果 C 的行数和列数。

* K 表示 GEMM 的 “reduction mode”，也就是 GEMM 求和的 mode。细节请参见 [GEMM tutorial](./0x_gemm_tutorial.md)。

我们使用 `(...) x (...) => (...)` 记法列出输入 `Tensor` A、B 和输出 `Tensor` C 的 modes。最左边两个 `(...)` 分别描述 A 和 B，`=>` 右侧的 `(...)` 描述 C。

1. `(V) x (V) => (V)`。向量逐元素乘积：C<sub>v</sub> += A<sub>v</sub> B<sub>v</sub>。dispatch 到 FMA 或 MMA。

2. `(M) x (N) => (M,N)`。向量外积：C<sub>mn</sub> += A<sub>m</sub> B<sub>n</sub>。以 V=1 dispatch 到第 4 种情况。

3. `(M,K) x (N,K) => (M,N)`。矩阵乘积：C<sub>mn</sub> += A<sub>mk</sub> B<sub>nk</sub>。对每个 K dispatch 到第 2 种情况。

4. `(V,M) x (V,N) => (V,M,N)`。向量 batched 外积：C<sub>vmn</sub> += A<sub>vm</sub> B<sub>vn</sub>。会优化 register reuse，并对每个 M、N dispatch 到第 1 种情况。

5. `(V,M,K) x (V,N,K) => (V,M,N)`。矩阵 batched 乘积：C<sub>vmn</sub> += A<sub>vmk</sub> B<sub>vnk</sub>。对每个 K dispatch 到第 4 种情况。

关于 CuTe mode 排序约定的概览，请参考 [GEMM tutorial](./0x_gemm_tutorial.md)。例如，如果 K 出现，它总是位于最右侧，也就是 “outermost”。如果 V 出现，它总是位于最左侧，也就是 “innermost”。

### Dispatch 到优化实现

和 `copy` 一样，CuTe 的 `gemm` 实现使用其 `Tensor` 参数的类型 dispatch 到适当优化的实现。同样和 `copy` 一样，`gemm` 接受可选的 `MMA_Atom` 参数，允许调用者覆盖 CuTe 基于 `Tensor` 参数类型默认选择的 `FMA` 指令。

关于 `MMA_Atom` 以及针对不同架构特化 `gemm` 的更多信息，请参考教程中的 [MMA 小节](./0t_mma_atom.md)。

## `axpby`

`axpby` 算法位于 [`include/cute/algorithm/axpby.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/axpby.hpp) 头文件中。它把 $\alpha x + \beta y$ 的结果赋给 $y$，其中 $\alpha$ 和 $\beta$ 是标量，$x$ 和 $y$ 是 `Tensor`。这个名字代表 “Alpha times X Plus Beta times Y”，是原始 BLAS “AXPY” 例程（“Alpha times X Plus Y”）的泛化。

## `fill`

`fill` 算法位于 [`include/cute/algorithm/fill.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/fill.hpp) 头文件中。它用给定标量值覆盖输出 `Tensor` 参数的元素。

## `clear`

`clear` 算法位于 [`include/cute/algorithm/clear.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/clear.hpp) 头文件中。它用 0 覆盖输出 `Tensor` 参数的元素。

## 其他算法

CuTe 还提供其他算法。它们的头文件可以在 [`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm) 目录中找到。

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
