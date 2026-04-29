# CuTe Layouts

本文档描述 `Layout`，也就是 CuTe 的核心抽象。从根本上说，`Layout` 将 coordinate space(s) 映射到 index space。

`Layout` 为多维数组访问提供统一接口，抽象掉数组元素在内存中如何组织的细节。这让用户可以通用地编写访问多维数组的算法；即使 layout 改变，用户代码也不需要改变。例如，row-major MxN layout 和 column-major MxN layout 可以在软件中以相同方式处理。

CuTe 还提供一套 “`Layout` algebra”。`Layout` 可以被组合和变换，用于构造更复杂 layout，也可以用一个 layout 去 tile 另一个 layout。这能帮助用户完成诸如把数据 layout 按线程 layout partition 之类的任务。

## 基础类型和概念

### Integers

CuTe 大量使用 dynamic（只在运行时已知）和 static（编译期已知）整数。

* Dynamic integers，也就是 “run-time integers”，只是普通整数类型，例如 `int`、`size_t` 或 `uint16_t`。任何被 `std::is_integral<T>` 接受的类型，在 CuTe 中都被视为 dynamic integer。

* Static integers，也就是 “compile-time integers”，是类似 `std::integral_constant<Value>` 的类型实例。这些类型把值编码为 `static constexpr` 成员。它们也支持转换到底层 dynamic 类型，因此可以与 dynamic integers 一起出现在表达式中。CuTe 定义了自己的 CUDA-compatible static integer 类型 `cute::C<Value>`，并重载数学 operators，使 static integers 上的数学运算仍产生 static integers。CuTe 还定义了便利别名 `Int<1>`、`Int<2>`、`Int<3>` 以及 `_1`、`_2`、`_3`，这些在示例中经常出现。

CuTe 尝试以相同方式处理 static 和 dynamic integers。后续示例中，所有 dynamic integers 都可以替换成 static integers，反之亦然。当我们在 CuTe 中说 “integer” 时，几乎总是指 static 或 dynamic integer。

CuTe 提供多个 traits 来处理 integers。

* `cute::is_integral<T>`：检查 `T` 是否为 static 或 dynamic integer 类型。
* `cute::is_std_integral<T>`：检查 `T` 是否为 dynamic integer 类型，等价于 `std::is_integral<T>`。
* `cute::is_static<T>`：检查 `T` 是否为空类型，也就是其实例不依赖任何 dynamic 信息，等价于 `std::is_empty`。
* `cute::is_constant<N,T>`：检查 `T` 是否为 static integer，并且其值等价于 `N`。

更多信息见 [`integral_constant` implementations](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric/integral_constant.hpp)。

### Tuple

tuple 是包含零个或多个元素的有限有序列表。[`cute::tuple` class](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container/tuple.hpp) 行为类似 `std::tuple`，但可在 device 和 host 上工作。它对模板参数施加限制，并为了性能和简洁性简化了实现。

### IntTuple

CuTe 将 IntTuple 概念定义为：一个 integer，或一个由 IntTuples 组成的 tuple。注意这是递归定义。在 C++ 中，CuTe 定义了 [operations on `IntTuple`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/int_tuple.hpp)。

`IntTuple` 的例子包括：

* `int{2}`，dynamic integer 2。
* `Int<3>{}`，static integer 3。
* `make_tuple(int{2}, Int<3>{})`，dynamic-2 和 static-3 的 tuple。
* `make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{})`，由 dynamic-42、static-1 与 dynamic-3 的 tuple、static-17 组成的 tuple。

CuTe 将 `IntTuple` 概念复用于许多不同事物，包括 Shape、Stride、Step 和 Coord，见 [`include/cute/layout.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp)。

定义在 `IntTuple` 上的操作包括：

* `rank(IntTuple)`：`IntTuple` 中元素数量。单个 integer 的 rank 是 1，tuple 的 rank 是 `tuple_size`。

* `get<I>(IntTuple)`：`IntTuple` 的第 `I` 个元素，其中 `I < rank`。对单个 integer，`get<0>` 就是该 integer。

* `depth(IntTuple)`：层次化 `IntTuple` 的数量。单个 integer depth 为 0；整数 tuple depth 为 1；包含整数 tuple 的 tuple depth 为 2，依此类推。

* `size(IntTuple)`：`IntTuple` 中所有元素的乘积。

我们用括号表示 `IntTuple` 的层次。例如，`6`、`(2)`、`(4,3)` 和 `(3,(6,2),8)` 都是 `IntTuple`。

### Shapes 和 Strides

`Shape` 和 `Stride` 都是 `IntTuple` 概念。

### Layout

`Layout` 是 (`Shape`, `Stride`) 组成的 tuple。语义上，它通过 `Stride` 实现从 `Shape` 内任意 coordinate 到 index 的映射。

### Tensor

`Layout` 可以与数据组合，例如 pointer 或 array，从而创建 `Tensor`。`Layout` 生成的 index 用于 subscript 一个 iterator 以取回合适的数据。关于 `Tensor` 的细节，请参考教程中的 [`Tensor` 小节](./03_tensor.md)。

## Layout 创建和使用

`Layout` 是一对 `IntTuple`：`Shape` 和 `Stride`。第一个元素定义 `Layout` 的抽象 *shape*，第二个元素定义 *strides*，也就是从 shape 内 coordinates 到 index space 的映射。

CuTe 在 `Layout` 上定义了很多与 `IntTuple` 类似的操作。

* `rank(Layout)`：`Layout` 中 modes 数量，等价于 `Layout` 的 shape 的 tuple size。

* `get<I>(Layout)`：`Layout` 的第 `I` 个 sub-layout，其中 `I < rank`。

* `depth(Layout)`：`Layout` shape 的 depth。单个 integer depth 为 0；整数 tuple depth 为 1；整数 tuples 的 tuple depth 为 2，依此类推。

* `shape(Layout)`：`Layout` 的 shape。

* `stride(Layout)`：`Layout` 的 stride。

* `size(Layout)`：`Layout` 函数 domain 的 size，等价于 `size(shape(Layout))`。

* `cosize(Layout)`：`Layout` 函数 codomain 的 size，不一定是 range，等价于 `A(size(A) - 1) + 1`。

### 层次化访问函数

`IntTuple` 和 `Layout` 可以任意嵌套。为了方便，CuTe 为上面某些函数定义了接受整数序列的版本，而不只是一个整数。这使得访问嵌套 `IntTuple` 或 `Layout` 内部元素更容易。例如，CuTe 允许 `get<I...>(x)`，其中 `I...` 是 “C++ parameter pack”，表示零个或多个整数模板参数。这些层次化访问函数包括：

* `get<I0,I1,...,IN>(x) := get<IN>(...(get<I1>(get<I0>(x)))...)`。提取 `x` 第 `I0` 个元素的第 `I1` 个元素的 ... 的第 `IN` 个元素。

* `rank<I...>(x) := rank(get<I...>(x))`。`x` 第 `I...` 个元素的 rank。

* `depth<I...>(x) := depth(get<I...>(x))`。`x` 第 `I...` 个元素的 depth。

* `shape<I...>(x) := shape(get<I...>(x))`。`x` 第 `I...` 个元素的 shape。

* `size<I...>(x) := size(get<I...>(x))`。`x` 第 `I...` 个元素的 size。

在下面的示例中，你会看到 `size<0>` 和 `size<1>` 被用于决定 layout 或 tensor 第 0、1 个 mode 的循环边界。

### 构造 Layout

`Layout` 可以用很多方式构造。它可以包含编译期 static integers 和运行时 dynamic integers 的任意组合。

```c++
Layout s8 = make_layout(Int<8>{});
Layout d8 = make_layout(8);

Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                             make_stride(Int<12>{},Int<1>{}));
Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                               LayoutLeft{});
Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                               LayoutRight{});

Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                           make_stride(4,make_stride(2,1)));
Layout s2xh4_col = make_layout(shape(s2xh4),
                               LayoutLeft{});
```

`make_layout` 函数返回一个 `Layout`。它推导函数参数类型，并返回带有合适模板参数的 `Layout`。类似地，`make_shape` 和 `make_stride` 分别返回 `Shape` 与 `Stride`。CuTe 经常使用这些 `make_*` 函数，因为 constructor template argument deduction（CTAD）有一些限制，并且可以避免重复写 static 或 dynamic integer 类型。

当省略 `Stride` 参数时，它会由提供的 `Shape` 生成，默认使用 `LayoutLeft`。`LayoutLeft` tag 会从左到右对 `Shape` 做 exclusive prefix product 来构造 strides，不考虑 `Shape` 的层次结构。这可以看作 “generalized column-major stride generation”。`LayoutRight` tag 会从右到左对 `Shape` 做 exclusive prefix product 来构造 strides，同样不考虑 `Shape` 的层次结构。对 depth 为 1 的 shapes，这可以看作 “row-major stride generation”；但对 hierarchical shapes，得到的 strides 可能令人意外。例如，上面的 `s2xh4` 的 strides 可以用 `LayoutRight` 生成。

对上面每个 layout 调用 `print` 会得到：

```text
s8        :  _8:_1
d8        :  8:_1
s2xs4     :  (_2,_4):(_1,_2)
s2xd4     :  (_2,4):(_1,_2)
s2xd4_a   :  (_2,4):(_12,_1)
s2xd4_col :  (_2,4):(_1,_2)
s2xd4_row :  (_2,4):(4,_1)
s2xh4     :  (2,(2,2)):(4,(2,1))
s2xh4_col :  (2,(2,2)):(_1,(2,4))
```

`Shape:Stride` 记法经常用于 `Layout`。`_N` 记法是 static integer 的简写，其他整数是 dynamic integers。注意，`Shape` 和 `Stride` 都可以由 static 和 dynamic integers 混合组成。

还要注意，`Shape` 和 `Stride` 被假定为 *congruent*。也就是说，`Shape` 和 `Stride` 具有相同 tuple profiles。`Shape` 中每个 integer 在 `Stride` 中都有对应 integer。可以用下面方式断言：

```cpp
static_assert(congruent(my_shape, my_stride));
```

### 使用 Layout

`Layout` 的基本用途是在 `Shape` 定义的 coordinate space(s) 与 `Stride` 定义的 index space 之间映射。例如，要以 2-D 表格打印任意 rank-2 layout，可以写：

```c++
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}
```

对上面示例会产生：

```text
> print2D(s2xs4)
  0    2    4    6
  1    3    5    7
> print2D(s2xd4_a)
  0    1    2    3
 12   13   14   15
> print2D(s2xh4_col)
  0    2    4    6
  1    3    5    7
> print2D(s2xh4)
  0    2    1    3
  4    6    5    7
```

这里可以看到 static、dynamic、row-major、column-major 和 hierarchical layouts。语句 `layout(m,n)` 给出逻辑 2-D coordinate `(m,n)` 到 1-D index 的映射。

有趣的是，`s2xh4` 示例既不是 row-major 也不是 column-major。而且它有三个 modes，却仍被解释为 rank-2，并且我们使用的是 2-D coordinate。具体来说，`s2xh4` 在第二个 mode 中有一个 2-D multi-mode，但我们仍然可以对该 mode 使用 1-D coordinate。下一节会继续说明。先再泛化一步：使用 1-D coordinate，并把每个 layout 的所有 modes 当作一个 single multi-mode。例如下面的 `print1D`：

```c++
template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}
```

会对上面示例产生：

```text
> print1D(s2xs4)
  0    1    2    3    4    5    6    7
> print1D(s2xd4_a)
  0   12    1   13    2   14    3   15
> print1D(s2xh4_col)
  0    1    2    3    4    5    6    7
> print1D(s2xh4)
  0    4    2    6    1    5    3    7
```

layout 的任何 multi-mode，包括整个 layout 本身，都可以接受 1-D coordinate。后续章节会详细说明。

CuTe 提供更多打印工具用于可视化 Layouts。`print_layout` 函数会生成 Layout 映射的格式化 2-D 表格。

```text
> print_layout(s2xh4)
(2,(2,2)):(4,(2,1))
      0   1   2   3
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+
```

`print_latex` 函数会生成 LaTeX，可以用 `pdflatex` 编译成同一个 2-D 表格的彩色矢量图。

### Vector Layouts

我们将任何 `rank == 1` 的 `Layout` 定义为 vector。例如，layout `8:1` 可以解释为一个 8 元素 vector，其 indices 连续。

```text
Layout:  8:1
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

类似地，layout `8:2` 可以解释为一个 8 元素 vector，其中元素 indices 以 `2` 为 stride。

```text
Layout:  8:2
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  8 10 12 14
```

根据上面的 rank-1 定义，我们也把 layout `((4,2)):((2,1))` 解释为 vector，因为它的 shape 是 rank-1。内部 shape 看起来像一个 4x2 row-major matrix，但额外的一对括号暗示我们可以把这两个 modes 解释成一个 1-D 8 元素 vector。strides 告诉我们，前 `4` 个元素 stride 为 `2`，然后有 `2` 份这样的元素，额外 stride 为 `1`。

```text
Layout:  ((4,2)):((2,1))
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  1  3  5  7
```

可以看到第二组 `4` 个元素是第一组 `4` 个元素的副本，只是额外 stride 为 `1`。

考虑 layout `((4,2)):((1,4))`。同样，它是 `4` 个元素 stride 为 `1`，然后有 `2` 份这样的元素，stride 为 `4`。

```text
Layout:  ((4,2)):((1,4))
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

作为从 integers 到 integers 的函数，它与 `8:1` 完全相同，是 identity function。

### Matrix examples

进一步泛化，我们将任何 rank-2 的 `Layout` 定义为 matrix。例如：

```text
Shape :  (4,2)
Stride:  (1,4)
  0   4
  1   5
  2   6
  3   7
```

这是一个 4x2 column-major layout，沿 columns 方向 stride-1，跨 rows 方向 stride-4。

```text
Shape :  (4,2)
Stride:  (2,1)
  0   1
  2   3
  4   5
  6   7
```

这是一个 4x2 row-major layout，沿 columns 方向 stride-2，跨 rows 方向 stride-1。Majorness 只是指哪个 mode 的 stride 是 1。

和 vector layouts 一样，matrix 的每个 mode 也可以被拆成 *multi-modes*。这让我们能表达 row-major 和 column-major 之外的更多 layouts。例如：

```text
Shape:  ((2,2),2)
Stride: ((4,1),2)
  0   2
  4   6
  1   3
  5   7
```

它在逻辑上也是 4x2，跨 rows 的 stride 为 2，但沿 columns 方向是 multi-stride。column 中前 `2` 个元素 stride 为 `4`，随后有一个 stride-1 的副本。由于该 layout 在逻辑上是 4x2，和上面的 column-major、row-major 示例一样，我们仍然可以使用 2-D coordinates 索引它。

## Layout 概念

本节介绍 `Layout` 接受的 coordinate sets，以及 coordinate mappings 和 index mappings 如何计算。

### Layout compatibility

如果 layout A 的 shape 与 layout B 的 shape compatible，则称 layout A 与 layout B *compatible*。Shape A 与 shape B compatible 的条件是：

* A 的 size 等于 B 的 size；
* A 内所有 coordinates 都是 B 内有效 coordinates。

例如：

* Shape 24 不 compatible with Shape 32。
* Shape 24 compatible with Shape (4,6)。
* Shape (4,6) compatible with Shape ((2,2),6)。
* Shape ((2,2),6) compatible with Shape ((2,2),(3,2))。
* Shape 24 compatible with Shape ((2,2),(3,2))。
* Shape 24 compatible with Shape ((2,3),4)。
* Shape ((2,3),4) 不 compatible with Shape ((2,2),(3,2))。
* Shape ((2,2),(3,2)) 不 compatible with Shape ((2,3),4)。
* Shape 24 compatible with Shape (24)。
* Shape (24) 不 compatible with Shape 24。
* Shape (24) 不 compatible with Shape (4,6)。

也就是说，*compatible* 是 Shapes 上的 weak partial order，因为它具有 reflexive、antisymmetric 和 transitive 性质。

### Layouts Coordinates

基于上面的 compatibility 概念，需要强调：每个 `Layout` 都接受多种 coordinates。每个 `Layout` 都接受任意与自身 compatible 的 `Shape` 所对应的 coordinates。CuTe 通过 colexicographical order 在这些 coordinate sets 之间提供映射。

因此，所有 Layouts 都提供两个基本映射：

* 从输入 coordinate 到通过 `Shape` 得到的对应 natural coordinate；
* 从 natural coordinate 到通过 `Stride` 得到的 index。

#### Coordinate Mapping

从输入 coordinate 到 natural coordinate 的映射是在 `Shape` 内应用 colexicographical order。Colexicographical 是从右到左读，而 lexicographical 是从左到右读。

以 shape `(3,(2,3))` 为例。这个 shape 有三种 coordinate sets：1-D coordinates、2-D coordinates 和 natural（h-D）coordinates。

| 1-D | 2-D | Natural | | 1-D | 2-D | Natural |
| --- | --- | --- |-| --- | --- | --- |
| `0` | `(0,0)` | `(0,(0,0))` | | `9` | `(0,3)` | `(0,(1,1))` |
| `1` | `(1,0)` | `(1,(0,0))` | | `10` | `(1,3)` | `(1,(1,1))` |
| `2` | `(2,0)` | `(2,(0,0))` | | `11` | `(2,3)` | `(2,(1,1))` |
| `3` | `(0,1)` | `(0,(1,0))` | | `12` | `(0,4)` | `(0,(0,2))` |
| `4` | `(1,1)` | `(1,(1,0))` | | `13` | `(1,4)` | `(1,(0,2))` |
| `5` | `(2,1)` | `(2,(1,0))` | | `14` | `(2,4)` | `(2,(0,2))` |
| `6` | `(0,2)` | `(0,(0,1))` | | `15` | `(0,5)` | `(0,(1,2))` |
| `7` | `(1,2)` | `(1,(0,1))` | | `16` | `(1,5)` | `(1,(1,2))` |
| `8` | `(2,2)` | `(2,(0,1))` | | `17` | `(2,5)` | `(2,(1,2))` |

进入 shape `(3,(2,3))` 的每个 coordinate 都有两个 *equivalent* coordinates，并且所有 equivalent coordinates 都映射到同一个 natural coordinate。再次强调，由于上面所有 coordinates 都是有效输入，Shape 为 `(3,(2,3))` 的 Layout 可以通过 1-D coordinates 被当作 18 元素一维数组使用，也可以通过 2-D coordinates 被当作 3x6 矩阵使用，或者通过 h-D natural coordinates 被当作 3x(2x3) tensor 使用。

前面的 1-D print 展示了 CuTe 如何用 2-D coordinates 的 colexicographical ordering 识别 1-D coordinates。从 `i = 0` 迭代到 `size(layout)`，并用单个 integer coordinate `i` 索引 layout，会以这种 “generalized-column-major” 顺序遍历 2-D coordinates，即使该 layout 把 coordinates 映射成 row-major 或更复杂的 indices。

函数 `cute::idx2crd(idx, shape)` 负责 coordinate mapping。它接受 shape 内任意 coordinate，并计算该 shape 的等价 natural coordinate。

```cpp
auto shape = Shape<_3,Shape<_2,_3>>{};
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```

#### Index Mapping

从 natural coordinate 到 index 的映射通过对 natural coordinate 和 `Layout` 的 `Stride` 做 inner product 完成。

以 layout `(3,(2,3)):(3,(12,1))` 为例。natural coordinate `(i,(j,k))` 会得到 index `i*3 + j*12 + k*1`。下方 2-D 表格展示这个 layout 计算出的 indices，其中 `i` 用作 row coordinate，`(j,k)` 用作 column coordinate。

```text
       0     1     2     3     4     5     <== 1-D col coord
     (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)   <== 2-D col coord (j,k)
    +-----+-----+-----+-----+-----+-----+
 0  |  0  |  12 |  1  |  13 |  2  |  14 |
    +-----+-----+-----+-----+-----+-----+
 1  |  3  |  15 |  4  |  16 |  5  |  17 |
    +-----+-----+-----+-----+-----+-----+
 2  |  6  |  18 |  7  |  19 |  8  |  20 |
    +-----+-----+-----+-----+-----+-----+
```

函数 `cute::crd2idx(c, shape, stride)` 负责 index mapping。它接受 shape 内任意 coordinate，计算该 shape 的等价 natural coordinate（如果它还不是 natural coordinate），并与 strides 做 inner product。

```cpp
auto shape  = Shape <_3,Shape<  _2,_3>>{};
auto stride = Stride<_3,Stride<_12,_1>>{};
print(crd2idx(   16, shape, stride));       // 17
print(crd2idx(_16{}, shape, stride));       // _17
print(crd2idx(make_coord(   1,   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},_5{}), shape, stride));  // _17
print(crd2idx(make_coord(   1,make_coord(   1,   2)), shape, stride));  // 17
print(crd2idx(make_coord(_1{},make_coord(_1{},_2{})), shape, stride));  // _17
```

## Layout 变换

### Sublayouts

可以用 `layout<I...>` 获取 sublayouts：

```cpp
Layout a   = Layout<Shape<_4,Shape<_3,_6>>>{}; // (4,(3,6)):(1,(4,12))
Layout a0  = layout<0>(a);                     // 4:1
Layout a1  = layout<1>(a);                     // (3,6):(4,12)
Layout a10 = layout<1,0>(a);                   // 3:4
Layout a11 = layout<1,1>(a);                   // 6:12
```

也可以用 `select<I...>`：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = select<1,3>(a);                   // (3,7):(2,30)
Layout a01 = select<0,1,3>(a);                 // (2,3,7):(1,2,30)
Layout a2  = select<2>(a);                     // (5):(6)
```

也可以用 `take<ModeBegin, ModeEnd>`：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = take<1,3>(a);                     // (3,5):(2,6)
Layout a14 = take<1,4>(a);                     // (3,5,7):(2,6,30)
// take<1,1> not allowed. Empty layouts not allowed.
```

### Concatenation

可以把 `Layout` 传给 `make_layout` 来 wrap 和 concatenate：

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout row = make_layout(a, b);                 // (3,4):(1,3)
Layout col = make_layout(b, a);                 // (4,3):(3,1)
Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
Layout aa  = make_layout(a);                    // (3):(1)
Layout aaa = make_layout(aa);                   // ((3)):((1))
Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
```

也可以用 `append`、`prepend` 或 `replace` 组合：

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout ab = append(a, b);                       // (3,4):(1,3)
Layout ba = prepend(a, b);                      // (4,3):(3,1)
Layout c  = append(ab, ab);                     // (3,4,(3,4)):(1,3,(1,3))
Layout d  = replace<2>(c, b);                   // (3,4,4):(1,3,3)
```

### Grouping 和 flattening

Layout modes 可以用 `group<ModeBegin, ModeEnd>` 分组，并用 `flatten` 展平。

```cpp
Layout a = Layout<Shape<_2,_3,_5,_7>>{};  // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout b = group<0,2>(a);                 // ((_2,_3),_5,_7):((_1,_2),_6,_30)
Layout c = group<1,3>(b);                 // ((_2,_3),(_5,_7)):((_1,_2),(_6,_30))
Layout f = flatten(b);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout e = flatten(c);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
```

Grouping、flattening 和 reordering modes 允许在原地重新解释 tensors，例如把 tensor 解释成 matrix、把 matrix 解释成 vector、把 vector 解释成 matrix 等。

### Slicing

`Layout` 可以被 sliced，但 slicing 更适合在 `Tensor` 上执行。Slicing 细节请见 [`Tensor` 小节](./03_tensor.md)。

## 总结

* `Layout` 的 `Shape` 定义它的 coordinate space(s)。

  * 每个 `Layout` 都有一个 1-D coordinate space。它可以用于按 colexicographical order 遍历 coordinate spaces。

  * 每个 `Layout` 都有一个 R-D coordinate space，其中 R 是 layout 的 rank。R-D coordinates 的 colexicographical enumeration 对应上面的 1-D coordinates。

  * 每个 `Layout` 都有一个 h-D natural coordinate space，其中 h 表示 hierarchical。这些 coordinates 按 colexicographical order 排序，该顺序的 enumeration 对应上面的 1-D coordinates。natural coordinate 与 `Shape` *congruent*，因此 coordinate 的每个元素在 `Shape` 中都有对应元素。

* `Layout` 的 `Stride` 将 coordinates 映射到 indices。

  * natural coordinate 的元素与 `Stride` 元素的 inner product 产生最终 index。

对每个 `Layout`，都存在一个与之 compatible 的 integral `Shape`，即 `size(layout)`。因此可以观察到：

> Layouts 是从 integers 到 integers 的函数。

如果你熟悉 C++23 的 `mdspan` 特性，这是 `mdspan` layout mappings 和 CuTe `Layout` 的重要区别。在 CuTe 中，`Layout` 是一等公民，原生支持层次化，可以自然表示 row-major 和 column-major 之外的函数，也可以用层次化 coordinates 索引。`mdspan` layout mappings 也能表示层次化函数，但这需要定义 custom layout。`mdspan` 的输入 coordinates 必须具有与 `mdspan` 相同的 shape；多维 `mdspan` 不接受 1-D coordinates。

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
