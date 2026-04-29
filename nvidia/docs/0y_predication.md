# Predication：当 tiling 不能完美整除时怎么办

[GEMM tutorial](./0x_gemm_tutorial.md) 展示了如何通过遍历输入矩阵和输出矩阵的 tiles 来计算 matrix-matrix multiply。示例都假设 tiles 能整除矩阵，没有 remainder。如果不是这样怎么办？例如，我们可能希望把一个 41 x 55 矩阵切成 4 x 8 tiles，但 41 / 4 余 1，55 / 8 余 7。矩阵中这些“剩余”部分应该如何处理？

首先注意，`logical_divide`（CuTe 的 layout tiling 方式）会“向上取整”。例如，如果 `N` 是 layout `1000:1`，`B` 是 layout `128:1`，那么 `logical_divide(N, B)` 是 layout `(128, 8):(1, 128)`。这实际上把原始 shape `N = 1000` 向上取整成一个 `128 x 8` 矩阵，就像 `N = 1024` 一样。那最后 24 个不属于原始数据的元素怎么办？最后一个 tile 如何处理，又如何避免越界索引？

和其他 CUDA 编程入门一样，CuTe 处理这些问题的惯用方法是 “predication”。CuTe 不尝试把 “remainder tiles” 表示成 “7 个 size-128 tiles 加 1 个 size-104 tile”，而是向上取整成 “8 个 size-128 tiles”，然后构造 predicates，让 kernel 只访问每个 tile 中在矩阵范围内有效的数据。这与 GPU 优化方式吻合：没有 warp divergence 的分支相对较快。它也符合常见 CUDA 写法：当把 N 个 work items 以一维方式分给 B 个 thread blocks 时，先检查 “my thread” 是否越界，再执行工作。

考虑一个通用 tiling：把 size-1000 vector 切成 size-128 chunks。可以如下构造 predication tensor：

```c++
Tensor gmem = ...     // e.g. size 1000
Tensor smem = ...     // e.g. size 128

// Tile the gmem for smem
Tensor gmem_tiled = logical_divide(gmem, size(smem));      // e.g. (128,8)

// Create an identity layout for gmem and tile it similarly
Layout id_layout = make_layout(shape(gmem));               // e.g. 1000:1, explicitly constructed as identity function
Layout id_tiled  = logical_divide(id_layout, size(smem));  // e.g. (128,8):(1,128), but many elements aren't "valid"

// Create a predicate tensor
Tensor pred = make_tensor<bool>(shape(id_tiled));          // e.g. (128,8)
for (int i = 0; i < size(pred); ++i) {
  pred(i) = id_tiled(i) < size(id_layout);  // Predicate: Is the offset within the original shape?
}

// ... intervening code ...

// Note that gmem_tiled, id_tiled, and pred tensors are all congruent
// For tile tile_i, determine if element value_j is in-bounds and copy to smem
if (pred(value_j,tile_i)) { smem(value_j) = gmem_tiled(value_j,tile_i); }
```

通用流程是：

1. 创建一个与原始数据 shape 相同的 “identity” layout，例如上例中的 `Layout id_layout = make_layout(shape(gmem))`；

2. 对这个 identity layout 重复相同的 tiling、partitioning、slicing，过程中可能向上取整，例如 `Layout id_tiled  = logical_divide(id_layout, size(smem));`；

3. 通过比较 reference layout 的坐标与原始 layout 的边界，创建 “predicate tensor”；

4. 使用 predicate tensor 屏蔽越界元素的访问。

作为相对简单的例子，考虑对 GEMM 的 epilogue 做 predication。假设我们已经把 `mC` partition 成 cta tiles，并按 mma 的线程进一步 partition：

```cpp
// CTA partitioning
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

// Thread partitioning
auto thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

// ... Compute gemms and accumulate into tCrC ...

// axpby epilogue
for (int i = 0; i < size(tCgC); ++i) {
  tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
}
```

按照 predication 流程写起来很直接：

```cpp
// A coordinate tensor the same shape as mC: (m,n) -> (m,n)
Tensor cC     = make_identity_tensor(shape(mC));

// Repeat partitioning steps applied to mC to our coordinate tensor cC
// CTA partitioning
Tensor cta_cC = local_tile(cC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N) -> (m,n)
// Thread partitioning
Tensor tCcC   = thr_mma.partition_C(cta_cC);                             // (MMA,MMA_M,MMA_N) -> (m,n)

// Predicated axpby epilogue
for (int i = 0; i < size(tCgC); ++i) {
  if (elem_less(tCcC(i), shape(mC))) {  // if coord is in-bounds
    tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
  }
}
```

上面，cta 负责对 `mC` 做 tiling/partitioning，mma 负责对 `gC` 做 tiling/partitioning，因此这两个步骤也都应用到 identity tensor 上。coordinate tensor `tCcC` 与 register fragment `tCrC` 以及 partitioned global memory tensor `tCgC` congruent，它们都是当前线程所拥有的数据 tile 的 subtensors。不过，`tCcC` tensor 在求值时仍保留原始 codomain：也就是原始 tensor `mC` 中的全局坐标。这个全局坐标会与 `mC` 的 shape 比较，以判断操作是否有效。

这种 “reference identity tensor” 或 “coordinate tensor” 方法的优点包括：

1. 它不依赖被 predicated tensor 的 layout/strides，只依赖逻辑边界。

2. partitioning 阶段可以是任意形式。CTA tiling、thread partitioning、TiledMMA 和 TiledCopy 都可以应用到任意 tensor，包括 coordinate tensor。

3. 它自然扩展到任意维度的 predication。

4. 它是典型 CUDA 一维并行 vector 访问模式的自然泛化：计算访问 index `idx`，并对访问 vector 的第 `idx` 个元素做 predication，以判断 `idx` 是否在边界内。

```cpp
int idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < N)  // idx is a "coord" into gmem and N is the "bound"
  gmem_ptr[idx] = ...;
```

在 SIMT 编程模型中，不应该修改 tensor extents 来避免循环越界。相反，predication 是一种通用方法：查询原始坐标，并判断该坐标是否越界。这样可以避免可变/动态 loop bounds，转而使用指令级 predication，保持线程一致性并维持负载均衡。它也足够通用，可以扩展到所有 ranks、所有线程和数据 layout，以及所有 tiling/partitioning patterns。针对特殊情况，可以把假设内建到 coordinate tensors 或 predicate tensors 中。

另一个稍复杂的例子是 GEMM 中 A 和 B loads 的 m- 与 n-predication。假设我们已经把 A 和 B tiles 按 cta 和 thread partition 如下：

```c++
// CTA partitioning
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

// Thread partitioning
Tensor tAgA = local_partition(gA, tA, thread_idx);                   // (THR_M,THR_K,k)
Tensor tAsA = local_partition(sA, tA, thread_idx);                   // (THR_M,THR_K)

Tensor tBgB = local_partition(gB, tB, thread_idx);                   // (THR_N,THR_K,k)
Tensor tBsB = local_partition(sB, tB, thread_idx);                   // (THR_N,THR_K)
```

`gA` 和 `gB` 分别是 `mA` 和 `mB` 根据 `cta_tiler` 与 `cta_coord` 得到的 tiles。`tAgA` 和 `tBgB` 分别是 `gA` 和 `gB` 根据 thread-layouts `tA`、`tB` 以及 `thread_idx` 得到的 partitions。

下面代码创建 “identity tensors”，它们映射坐标 `(m,k) -> (m,k)` 和 `(n,k) -> (n,k)`。

```c++
// Coordinate tensors
Tensor cA = make_identity_tensor(shape(mA));   // (m,k) -> (m,k)
Tensor cB = make_identity_tensor(shape(mB));   // (n,k) -> (n,k)
```

然后，以和 `mA`、`mB` 被 tiled/partitioned 成 `tAgA`、`tBgB` 完全相同的方式，对 reference tensors 做 tile 和 partition。

```c++
// CTA partitioning
Tensor cta_cA = local_tile(cA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k) -> (m,k)
Tensor cta_cB = local_tile(cB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k) -> (n,k)

// Thread partitioning
Tensor tAcA = local_partition(cta_cA, tA, thread_idx);                   // (THR_M,THR_K,k) -> (m,k)
Tensor tBcB = local_partition(cta_cB, tB, thread_idx);                   // (THR_N,THR_K,k) -> (m,k)
```

下面代码创建与 `tAgA` 和 `tBgB` 对应的 predicate tensors。它们会在 prologue 中计算一次，并用于在 inner loop 中屏蔽指令。

```c++
Tensor tApA = make_tensor<bool>(make_shape (size<0>(tAcA), size<1>(tAcA)),
                                make_stride(     Int<1>{},      Int<0>{}));
Tensor tBpB = make_tensor<bool>(make_shape (size<0>(tBcB), size<1>(tBcB)),
                                make_stride(     Int<1>{},      Int<0>{}));
```

这里做了几个假设：我们一次只关心一个 data tile 的 predicates，并且只关心 m- 和 n-modes 的 predicates，而 k-mode predicates 会用其他方式处理。m- 和 n-predicates 会被视为在每个 tile 中都保持不变，并在 mainloop 的每次迭代中复用。因此，我们只存储 m- 和 n-modes 的 predicates，并通过 stride-0 在 k-mode 上广播它们。填充 tensors 时也沿用同一假设：

```c++
// Populate the m- and n-predicates
CUTE_UNROLL
for (int m = 0; m < size<0>(tApA); ++m) {
  tApA(m,0) = elem_less(get<0>(tAcA(m,0,0)), shape<0>(mA));  // Compare the m-coordinate
}
CUTE_UNROLL
for (int n = 0; n < size<0>(tBpB); ++n) {
  tBpB(n,0) = elem_less(get<0>(tBcB(n,0,0)), shape<0>(mB));  // Compare the n-coordinate
}
```

这里仅比较第 0 个 k-tile 和第 0 个 k-block 的 m- 与 n-coordinates。stride-0 broadcasting mode 仍允许我们把这份数据当作每个待加载 tile 元素的 predicate tensor。

最后，就可以在 `copy_if` 中使用这些 predicate tensors，只复制对应 predicate tensor 元素为 `true` 的元素。

```c++
// Copy a k_tile from global memory to shared memory
copy_if(tApA, tAgA(_,_,k_tile), tAsA);
copy_if(tBpB, tBgB(_,_,k_tile), tBsB);
```
