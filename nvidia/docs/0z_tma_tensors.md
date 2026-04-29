# CuTe TMA Tensors

在使用 CuTe 的过程中，你可能会看到一些打印结果很奇怪的 CuTe Tensors，例如：

```text
ArithTuple(0,_0,_0,_0) o ((_128,_64),2,3,1):((_1@0,_1@1),_64@1,_1@2,_1@3)
```

什么是 `ArithTuple`？那些是 tensor strides 吗？它们是什么意思？它们有什么用途？

本文档旨在回答这些问题，并介绍 CuTe 的一些更高级特性。

## TMA 指令简介

Tensor Memory Accelerator（TMA）是一组用于在 global memory 和 shared memory 之间复制可能多维数组的指令。TMA 在 Hopper 架构中引入。一条 TMA 指令可以一次性复制整个数据 tile。因此，硬件不再需要为 tile 中每个元素分别计算内存地址并发出单独的 copy 指令。

为实现这一点，TMA 指令会接收一个 *TMA descriptor*。它是 global memory 中一个 1、2、3、4 或 5 维多维 tensor 的 packed 表示。TMA descriptor 保存：

* tensor 的 base pointer；

* tensor 元素的数据类型，例如 `int`、`float`、`double` 或 `half`；

* 每个维度的 size；

* 每个维度内的 stride；

* 其他 flags，用于表示 smem box size、smem swizzling patterns 和 out-of-bounds access behavior。

这个 descriptor 必须在 kernel 执行前于 host 上创建。所有会发出 TMA 指令的 thread blocks 都共享它。进入 kernel 后，TMA 会以以下参数执行：

* 指向 TMA descriptor 的 pointer；

* 指向 SMEM 的 pointer；

* 进入 TMA descriptor 所表示 GMEM tensor 的 coordinates。

例如，带 3-D coordinates 的 TMA-store 接口如下：

```cpp
struct SM90_TMA_STORE_3D {
  CUTE_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2) {
    // ... invoke CUDA PTX instruction ...
  }
};
```

可以看到，TMA 指令并不直接消费指向 global memory 的 pointer。global memory pointer 包含在 descriptor 中，被视为常量，并且不是 TMA 指令的独立参数。相反，TMA 消费的是进入 TMA global memory 视图的 TMA coordinates，而这个视图由 TMA descriptor 定义。

这意味着：普通 CuTe Tensor 存储 GMEM pointer，并计算 offsets 和新的 GMEM pointers；这种普通 Tensor 对 TMA 来说并没有用。

那应该怎么办？

## 构建 TMA Tensor

### 隐式 CuTe Tensors

所有 CuTe Tensors 都是 Layouts 和 Iterators 的组合。普通 global memory tensor 的 iterator 是它的 global memory pointer。不过，CuTe Tensor 的 iterator 不一定必须是 pointer；它可以是任何 random-access iterator。

这种 iterator 的一个例子是 *counting iterator*。它表示一个可能无限的整数序列，从某个值开始。我们把这个序列的成员称为 *implicit integers*，因为序列没有显式存储在 memory 中。iterator 只存储当前值。

我们可以使用 counting iterator 创建一个 implicit integers 的 tensor：

```cpp
Tensor A = make_tensor(counting_iterator<int>(42), make_shape(4,5));
print_tensor(A);
```

输出：

```text
counting_iter(42) o (4,5):(_1,4):
   42   46   50   54   58
   43   47   51   55   59
   44   48   52   56   60
   45   49   53   57   61
```

这个 tensor 将逻辑 coordinates 映射到即时计算出来的 integers。因为它仍然是 CuTe Tensor，所以它仍然可以像普通 tensor 一样被 tiled、partitioned 和 sliced，只是这些操作会把 integer offsets 累加到 iterator 中。

但是 TMA 不消费 pointers 或 integers，它消费 coordinates。我们能不能构造一个 implicit TMA coordinates 的 tensor，让 TMA 指令消费？如果可以，那么我们理论上也可以 tile、partition 和 slice 这个 coordinates tensor，从而总是得到正确的 TMA coordinate 并传给指令。

### ArithTupleIterators 和 ArithTuples

首先，我们为 TMA coordinates 构造一个类似 `counting_iterator` 的东西。它应该支持：

* 解引用得到一个 TMA coordinate；

* 被另一个 TMA coordinate offset。

我们称它为 `ArithmeticTupleIterator`。它存储一个 coordinate，也就是整数 tuple，并将其表示为 `ArithmeticTuple`。`ArithmeticTuple` 本质上是 `cute::tuple` 的 public subclass，只是重载了 `operator+`，使它能被另一个 tuple offset。两个 tuples 的和就是元素逐项相加后的 tuple。

现在，类似 `counting_iterator<int>(42)`，我们可以创建一个隐式 tuple “iterator”。它没有 increment 或其他常见 iterator 操作，但可以解引用，也可以被其他 tuples offset：

```cpp
ArithmeticTupleIterator citer_1 = make_inttuple_iter(42, Int<2>{}, Int<7>{});
ArithmeticTupleIterator citer_2 = citer_1 + make_tuple(Int<0>{}, 5, Int<2>{});
print(*citer_2);
```

输出：

```text
(42,7,_9)
```

TMA Tensor 可以使用这种 iterator 来存储当前 TMA coordinate “offset”。这里的 “offset” 加引号，是因为它显然不是普通的一维数组 offset 或 pointer。

总结一下：我们为 *整个 global memory tensor* 创建 TMA descriptor。TMA descriptor 定义该 tensor 的一个视图，指令接收进入该视图的 TMA coordinates。为了生成和跟踪这些 TMA coordinates，我们定义一个 implicit CuTe Tensor，它保存 TMA coordinates，并且能以和普通 CuTe Tensor 完全相同的方式被 tiled、sliced 和 partitioned。

现在我们可以用这个 iterator 跟踪并 offset TMA coordinates，但如何让 CuTe Layouts 生成非整数 offsets？

### Strides 不只是整数

普通 tensor 有一个 layout，将逻辑 coordinate `(i,j)` 映射到一维线性 index `k`。这个映射是 coordinate 与 strides 的 inner-product。

TMA Tensors 持有 TMA coordinates 的 iterators。因此，TMA Tensor 的 Layout 必须把逻辑 coordinate 映射到 TMA coordinate，而不是一维线性 index。

为此，我们可以抽象 stride 的含义。Stride 不一定必须是整数，而可以是任何支持与整数做 inner-product 的代数对象。显然的选择是前面用过的 `ArithmeticTuple`，因为它们可以互相相加；这一次还额外配备 `operator*`，使它们可以被整数缩放。

#### 补充：Integer-module strides

一组支持元素之间加法，以及元素与整数之间乘法的对象称为 integer-module。

形式上，integer-module 是一个 abelian group `(M,+)`，并配备 `Z*M -> M`，其中 `Z` 是整数。也就是说，integer-module `M` 是支持与整数做 inner products 的 group。整数本身是 integer-module。rank-R 的整数 tuples 也是 integer-module。

原则上，layout strides 可以是任意 integer-module。

#### Basis elements

CuTe 的 basis elements 位于 `cute/numeric/arithmetic_tuple.hpp` 头文件。为了便于创建可作为 strides 的 `ArithmeticTuple`，CuTe 使用 `E` 类型别名定义 normalized basis elements。“Normalized” 表示 basis element 的 scaling factor 是编译期整数 1。

| C++ object | Description | String representation |
| --- | --- | --- |
| `E<>{}` | `1` | `1` |
| `E<0>{}` | `(1,0,...)` | `1@0` |
| `E<1>{}` | `(0,1,0,...)` | `1@1` |
| `E<0,0>{}` | `((1,0,...),0,...)` | `1@0@0` |
| `E<0,1>{}` | `((0,1,0,...),0,...)` | `1@1@0` |
| `E<1,0>{}` | `(0,(1,0,...),0,...)` | `1@0@1` |
| `E<1,1>{}` | `(0,(0,1,0,...),0,...)` | `1@1@1` |

上表中的 “description” 列把每个 basis element 解释成无限整数 tuple，其中没有由 element 类型指定的 tuple entries 都是 0。我们从左到右计数 tuple entries，从 0 开始。例如，`E<1>{}` 在位置 1 上有一个 1：`(0,1,0,...)`。`E<3>{}` 在位置 3 上有一个 1：`(0,0,0,1,0,...)`。

Basis elements 可以是 *nested*。例如，上表中的 `E<0,1>{}` 表示位置 0 上有一个 `E<1>{}`：`((0,1,0,...),0,...)`。类似地，`1@1@0` 表示先把 `1` 提升到位置 1 形成 `1@1`：`(0,1,0,...)`，然后再把它提升到位置 0。

Basis elements 可以被 *scaled*，也就是乘以一个整数 *scaling factor*。例如，在 `5*E<1>{}` 中，scaling factor 是 `5`。`5*E<1>{}` 打印为 `5@1`，表示 `(0,5,0,...)`。scaling factor 会穿过任意 nesting。例如，`5*E<0,1>{}` 打印为 `5@1@0`，表示 `((0,5,0,...),0,...)`。

Basis elements 也可以相加，只要它们的层次结构兼容。例如，`3*E<0>{} + 4*E<1>{}` 结果是 `(3,4,0,...)`。直觉上，“compatible” 表示两个 basis elements 的嵌套结构足够匹配，可以相加。

#### Strides 的线性组合

Layouts 的工作方式是对自然 coordinate 与 strides 做 inner product。对于由整数元素组成的 strides，例如 `(1,100)`，输入 coordinate `(i,j)` 与 stride 的 inner product 是 `i + 100j`。用这个 index offset 一个“普通” tensor 的 pointer，就能得到 tensor 在 `(i,j)` 处元素的 pointer。

对于由 basis elements 组成的 strides，我们仍然计算自然 coordinate 与 strides 的 inner product。例如，如果 stride 是 `(1@0,1@1)`，那么输入 coordinate `(i,j)` 与 strides 的 inner product 是 `i@0 + j@1 = (i,j)`。这就转换成 TMA coordinate `(i,j)`。如果我们想反转 coordinates，可以使用 `(1@1,1@0)` 作为 stride。求值 layout 会得到 `i@1 + j@0 = (j,i)`。

Basis elements 的线性组合可以解释为可能多维、可能层次化的 coordinate。例如，`2*2@1@0 + 3*1@1 + 4*5@1 + 7*1@0@0` 表示 `((0,4,...),0,...) + (0,3,0,...) + (0,20,0,...) + ((7,...),...) = ((7,4,...),23,...)`，可以解释成 coordinate `((7,4),23)`。

因此，这些 strides 的线性组合可以用于生成 TMA coordinates。随后，这些 coordinates 又可以用于 offset TMA coordinate iterators。

### 应用于 TMA Tensors

现在我们可以构建引言中看到的那类 CuTe Tensors。

```cpp
Tensor a = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<0>{}, E<1>{}));
print_tensor(a);

Tensor b = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<1>{}, E<0>{}));
print_tensor(b);
```

打印结果：

```text
ArithTuple(0,0) o (4,5):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)

ArithTuple(0,0) o (4,5):(_1@1,_1@0):
  (0,0)  (1,0)  (2,0)  (3,0)  (4,0)
  (0,1)  (1,1)  (2,1)  (3,1)  (4,1)
  (0,2)  (1,2)  (2,2)  (3,2)  (4,2)
  (0,3)  (1,3)  (2,3)  (3,3)  (4,3)
```

### Copyright

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
