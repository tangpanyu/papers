# Bischof 和 Van Loan - 1987 - The WY Representation for Products of Householder Matrices

## 论文一句话总结

这篇论文提出用 WY 表示把多个 Householder 反射矩阵的乘积压成

$$
Q = I + WY^\top
$$

从而把传统 Householder QR 中大量矩阵-向量更新，改写成适合高性能机器的块状矩阵-矩阵计算，同时基本保留 Householder 方法的数值稳定性。

它的本质贡献是：不牺牲 Householder QR 的稳定性，把算法的主要工作从 BLAS-2 风格搬到 BLAS-3 风格。

## 1. 背景和问题

Householder QR 是经典、稳定的 QR 分解方法。给定矩阵 $A\in\mathbb{R}^{m\times n}$，它通过一系列 Householder 反射 $P_1,P_2,\ldots,P_n$ 把 $A$ 化成上三角矩阵 $R$：

$$
P_n\cdots P_2P_1A = R
$$

问题是，传统实现里每次反射通常以矩阵-向量操作为主，例如内积和 SAXPY。这样的计算在数学上很优雅，但在高性能硬件上容易被内存访问拖慢。

论文写作年代的 FPS-164/MAX、Cray-2、IBM 3090 等机器已经显示出一个趋势：真正快的不是零散向量操作，而是可复用数据、可流水化、可并行的矩阵-矩阵操作。

LU、Cholesky 这类分解天然容易块化；但正交变换，尤其 Householder QR，块化没那么直接。本文要解决的就是：如何把稳定的 Householder QR 组织成块算法。

## 2. 核心结论

论文的核心结论有四个：

- 任意一串 Householder 矩阵的乘积 $Q_k=P_1P_2\cdots P_k$ 可以表示为 $Q_k=I+W_kY_k^\top$。
- 有了 WY 表示，应用一批反射到矩阵 $B$ 上可以写成 $B\leftarrow B+W(Y^\top B)$ 或 $B\leftarrow B+Y(W^\top B)$，主要变成矩阵-矩阵乘法。
- 用块列 panel 组织 QR 时，只需要为当前 panel 保存 $W,Y$，不需要保存全局 $m\times n$ 的 $W$。
- 附录给出舍入误差分析，说明 WY 形式的操作仍具有和传统 Householder 类似的稳定性。

代价是生成 WY 因子本身仍有一定额外开销；但当矩阵足够大、块数足够多时，这部分相对 trailing matrix update 很小。真正的收益来自数据复用和矩阵-矩阵计算形态。

## 3. Mental Model

可以把传统 Householder QR 想成“每来一个反射，就立刻拿它去刷一遍后面的矩阵”。每个反射只带一个向量，所以更新很细碎：读一列、做内积、做 rank-1 更新，再读下一列。

WY 表示像是“先把一组反射打包成一个小工具箱，再一次性拿这个工具箱去刷后面的大矩阵”。工具箱就是 $W$ 和 $Y$。单个反射是：

$$
P_i = I + u_iv_i^\top
$$

一组反射合起来变成：

$$
Q = P_1P_2\cdots P_k = I + WY^\top
$$

这样更新右侧矩阵块 $B$ 时，不再逐个反射扫：

$$
B \leftarrow P_k^\top\cdots P_1^\top B
$$

而是做两步：

$$
Z = W^\top B
$$

$$
B \leftarrow B + YZ
$$

这里 $W,Y\in\mathbb{R}^{r\times p}$，$B\in\mathbb{R}^{r\times q}$，$Z\in\mathbb{R}^{p\times q}$。如果 $p$ 是 block size，$q$ 是右侧剩余列数，那么 $W^\top B$ 和 $YZ$ 都是典型的块状矩阵乘法。

真正移动的瓶颈是访存和数据复用。传统 QR 每个 Householder 向量被用于很多列，但操作粒度小；WY QR 把一组向量放进 $W,Y$，让它们在更新整个 trailing matrix 时被重复使用。

如果只记一张图，应该记住这个数据流：

$$
\text{panel } A_k
\rightarrow
\text{Householder vectors}
\rightarrow
W,Y
\rightarrow
Z=W^\top B
\rightarrow
B\leftarrow B+YZ
$$

其中 panel 负责生成反射，trailing matrix update 负责消耗大部分计算量。

## 4. 方法总览

### 4.1 标准 Householder 矩阵

论文采用的 Householder 写法是：

$$
P = I + uv^\top,\quad u=-2v,\quad \|v\|_2=1
$$

这和常见写法 $P=I-2vv^\top$ 等价，因为 $u=-2v$。

传统 QR 每一步构造一个 $P_k$，把当前列的下方元素消成 0。最后得到：

$$
Q^\top A = R
$$

或等价地：

$$
A = QR
$$

### 4.2 WY 表示

设已经有前 $k-1$ 个反射的乘积：

$$
Q_{k-1}=I+W_{k-1}Y_{k-1}^\top
$$

新反射为：

$$
P_k=I+u_kv_k^\top
$$

则：

$$
Q_k
=Q_{k-1}P_k
=(I+W_{k-1}Y_{k-1}^\top)(I+u_kv_k^\top)
$$

展开后可以得到一种更新方式：

$$
W_k=[W_{k-1},\ Q_{k-1}u_k]
$$

$$
Y_k=[Y_{k-1},\ v_k]
$$

这就是论文的 Method 1。

另一种方式是：

$$
W_k=[W_{k-1},\ u_k]
$$

$$
Y_k=[P_kY_{k-1},\ v_k]
$$

这是 Method 2。论文实现倾向 Method 1，因为 $Q_{k-1}u_k$ 更容易优化；但作者也指出 Method 2 在 QR 里可能让 $W,Y$ 都保持下梯形结构，存储上更漂亮。

### 4.3 为什么能加速

如果 $Q=I+WY^\top$，那么对矩阵 $B$ 应用 $Q$：

$$
QB = B + W(Y^\top B)
$$

对矩阵 $B$ 应用 $Q^\top$：

$$
Q^\top B = B + Y(W^\top B)
$$

这两个式子都把主要计算变成了矩阵乘法。相比逐个 Householder 反射，这种写法有更好的数据局部性，也更容易使用并行点积、并行 SAXPY 或现代 BLAS-3 GEMM。

## 5. 关键公式 / 算法

### 5.1 Householder 乘积的 WY 递推

核心公式：

$$
Q_k=Q_{k-1}P_k
=I+[W_{k-1},\ Q_{k-1}u_k][Y_{k-1},\ v_k]^\top
$$

作用：把第 $k$ 个 Householder 反射合并进已有 WY 因子。

符号和 shape：

- $Q_k\in\mathbb{R}^{m\times m}$：前 $k$ 个 Householder 的乘积。
- $W_k,Y_k\in\mathbb{R}^{m\times k}$：WY 因子。
- $u_k,v_k\in\mathbb{R}^{m}$：第 $k$ 个 Householder 的向量。
- $P_k=I+u_kv_k^\top\in\mathbb{R}^{m\times m}$。

直觉：$W,Y$ 的每一列记录一个反射带来的新增方向，但新增的 $u_k$ 需要先被前面累计的 $Q_{k-1}$ 变换一下，才能正确表示乘积顺序。

极小例子：如果有两个反射 $P_1,P_2$：

$$
Q_2=P_1P_2=(I+u_1v_1^\top)(I+u_2v_2^\top)
$$

则：

$$
W_2=[u_1,\ P_1u_2],\quad Y_2=[v_1,\ v_2]
$$

于是：

$$
Q_2=I+W_2Y_2^\top
$$

简单 PyTorch 语义：

```python
# W, Y: [m, k-1]
# u, v: [m]
# Q_u = (I + W @ Y.T) @ u
Q_u = u + W @ (Y.T @ u)
W = torch.cat([W, Q_u[:, None]], dim=1)
Y = torch.cat([Y, v[:, None]], dim=1)
```

### 5.2 用 WY 表示应用一组反射

如果当前 panel 生成了 $Q=I+WY^\top$，要把 $Q^\top$ 应用到右侧矩阵 $B$：

$$
Q^\top B=(I+YW^\top)B = B + Y(W^\top B)
$$

可以分两步：

$$
Z = W^\top B
$$

$$
B \leftarrow B + YZ
$$

shape：

- $W,Y\in\mathbb{R}^{r\times p}$。
- $B\in\mathbb{R}^{r\times q}$。
- $Z\in\mathbb{R}^{p\times q}$。

这里 $r$ 是当前剩余行数，$p$ 是 panel 宽度，$q$ 是 trailing matrix 的剩余列数。

直觉：$W^\top B$ 先把右侧矩阵投影到这一组 Householder 方向上，$YZ$ 再把这组方向的整体影响加回 $B$。它像“先汇总这一组反射对所有列的影响，再批量更新所有列”。

简单 PyTorch 语义：

```python
# W, Y: [r, p]
# B: [r, q]
Z = W.T @ B
B = B + Y @ Z
```

### 5.3 块 WY QR

论文真正推荐的是块算法。把 $A$ 按列分成：

$$
A=[A_1,A_2,\ldots,A_N]
$$

每个 block/panel 有 $p$ 列。第 $k$ 步关注：

$$
A(s:m,\ s:n)=[A_k,\ B]
$$

其中 $A_k$ 是当前 panel，$B$ 是右侧 trailing matrix。

算法：

1. 在当前 panel $A_k$ 上执行普通 Householder QR，并生成对应的 $W,Y$。
2. 用 WY 因子更新右侧矩阵：

$$
A(s:m,\ s:n)\leftarrow (I+WY^\top)^\top A(s:m,\ s:n)
$$

也就是对右侧 $B$ 做：

$$
B\leftarrow B+Y(W^\top B)
$$

论文指出，生成 $W,Y$ 的计算量约为：

$$
\frac{2mn^2-n^3}{N}
$$

应用 WY 因子的计算量约为：

$$
mn^2-\frac{n^3}{3}
$$

传统 LINPACK QR 的主要 flop 数也是：

$$
mn^2-\frac{n^3}{3}
$$

所以块 WY QR 从 flop 角度大约贵一个因子：

$$
1+\frac{2}{N}
$$

当 block 数 $N$ 较大时，这个额外成本接近可以忽略。更重要的是，高性能机器上 flop 数不是唯一指标，内存访问和数据复用往往更关键。

## 6. 数据流和实现设计

### 6.1 整体数据流

输入：

$$
A\in\mathbb{R}^{m\times n}
$$

选择 block width $p$，将列分块：

$$
A=[A_1,\ldots,A_N],\quad A_k\in\mathbb{R}^{m\times p}
$$

第 $k$ 步：

1. 定位当前剩余子矩阵 $A(s:m,s:n)$。
2. 对当前 panel $A(s:m,s:s+p-1)$ 做 Householder QR。
3. 在 panel 内累计生成 $W,Y\in\mathbb{R}^{r\times p}$。
4. 对右侧 trailing matrix $B=A(s:m,s+p:n)$ 做：

$$
Z=W^\top B
$$

$$
B=B+YZ
$$

输出：

- $A$ 的上三角部分被覆盖为 $R$。
- Householder 向量 $v_i$ 可以像传统 QR 一样存在对角线下方。
- $W$ 不需要跨 step 保存，只需要当前 panel 的工作区。

### 6.2 工作区

朴素 WY QR 需要保存全局 $W\in\mathbb{R}^{m\times n}$，这太贵。

块 WY QR 只需要当前 panel 的：

$$
W\in\mathbb{R}^{m\times p}
$$

因此工作区从 $m\times n$ 降到 $m\times p$。当 $p\ll n$ 时，这个差异很大。

### 6.3 FPS-164/MAX 上的实现含义

论文针对 FPS-164/MAX 做了实现。该机器的 MAX board 擅长两类操作：

- 并行 dot product。
- 并行 SAXPY。

对应到 WY update：

第一步：

$$
Z^\top = B^\top W
$$

这是一批并行点积。$W$ 可以装入 MAX vector registers，然后对 $B$ 的多列重复使用。

第二步：

$$
B\leftarrow B+YZ
$$

这是批量 SAXPY。把 $B$ 分块装入寄存器后，用 $Y$ 的列和 $Z$ 的系数重复更新。

所以 WY 表示非常贴合该机器的优势：$W$ 和 $Y$ 被加载后会被重复使用很多次，load/unload 成本被摊薄。

### 6.4 现代视角

用今天的术语看，这篇论文是在把 Householder QR 的 trailing update 变成 BLAS-3 kernel：

```python
Z = W.T @ B
B = B + Y @ Z
```

现代 LAPACK 中常见的 blocked QR / compact WY / WY representation，背后的思想和这篇论文高度一致。区别是现代实现通常会使用更标准的紧凑形式，例如 $I - VT V^\top$，并把三角因子 $T$ 显式维护出来，方便调用 GEMM。

从 kernel 角度看，这篇论文的核心目标不是减少数学 flop，而是提升 arithmetic intensity：让每次从内存读入的数据参与更多浮点运算。

## 7. 实验和效果

论文在 Cornell 的 FPS-164/MAX 上测试了 WY QR。

硬件背景：

- FPS-164 是 64-bit 科学处理器，峰值约 11 Mflops。
- 每块 MAX board 增加约 22 Mflops。
- 测试系统有 1 块 MAX board。
- 该配置下并行点积峰值约 33 Mflops，并行 SAXPY 峰值约 15 Mflops。

实验中，矩阵规模 $m$ 取 250、500、750、1000，$n$ 分别取 $0.25m,0.5m,0.75m,m$。

结果大致如下：

| $m$ | $n=0.25m$ | $n=0.5m$ | $n=0.75m$ | $n=m$ |
|---:|---:|---:|---:|---:|
| 250 | 7 | 9 | 11 | 11 |
| 500 | 10 | 13 | 14 | 15 |
| 750 | 13 | 14 | 16 | 17 |
| 1000 | 14 | 15 | 17 | 18 |

单位是 Mflops。

论文提到，优化过的传统 LINPACK QR 在没有 MAX board 的 FPS-164 上约为 6 Mflops。WY QR 的结果说明：虽然块 WY QR 可能多做少量 flop，但由于更多计算运行在 MAX board 喜欢的并行 dot product / SAXPY 形态上，整体性能显著更好。

这些实验的意义不是证明某个绝对 Mflops 数字今天仍有价值，而是证明一个算法设计原则：高性能线性代数不只看 flop 数，还要看计算能否变成硬件喜欢的密集块操作。

## 8. 稳定性

论文附录给出一个 $O(u)$ 级别的舍入误差分析。这里 $u$ 是 unit roundoff。

作者定义，如果：

$$
Q=I+WY^\top
$$

满足：

$$
\|W\|_2=O(1)
$$

$$
\|Y\|_2=O(1)
$$

$$
\|Q^\top Q-I\|_2=O(u)
$$

则称它是 $u$-orthogonal。

在这个条件下，用浮点数计算 $QA$ 可以写成：

$$
\mathrm{fl}(QA)=Q(A+E)
$$

其中：

$$
\|E\|_2=O(u)\|A\|_2
$$

这就是后向稳定性的味道：计算结果等价于对一个轻微扰动过的输入矩阵精确应用 $Q$。

论文进一步说明，Method 1 和 Method 2 的 WY 更新都会保持这种性质。因此，用 WY 形式组织 Householder 计算不会破坏传统 Householder 方法的良好数值性质。

## 9. 和其他分解的关系

作者也讨论了 WY 表示能否用于其他 Householder 型分解，例如：

$$
Q^\top AQ=T
$$

其中 $A$ 是对称矩阵，$T$ 是三对角矩阵；

$$
Q^\top AQ=H
$$

其中 $H$ 是 Hessenberg 矩阵；

$$
U^\top AV=B
$$

其中 $B$ 是双对角矩阵。

这些问题比 QR 更难块化，因为它们通常需要同时从左边和右边应用变换。QR 的特殊优势在于，Householder 只从一侧作用，所以可以先处理当前 panel，再延迟更新右侧矩阵。

对称三对角化则不同：下一步 Householder 的构造依赖已经从两侧更新后的矩阵列。如果延迟更新，可能无法正确得到下一列。作者认为直接聚合 Householder 并不划算，建议考虑块三对角化等替代路线。

## 10. 我的判断

真正贡献：这篇论文把 Householder 反射的乘积变成可维护的低秩更新形式，并展示了如何用它构造高性能块 QR。它是后来 blocked Householder QR、compact WY representation、BLAS-3 QR 实现的重要基础之一。

值得精读：

- 第 3 节：WY 表示的存在性和两种更新方式。
- 第 4 节：块 QR 算法，这是论文最核心的工程设计。
- 附录：稳定性分析，理解为什么这种重排不破坏 Householder 的数值优势。

可以略读：

- 第 5 节里 FPS-164/MAX 的硬件细节。历史意义很大，但现代读者只需要抓住“数据复用、并行 dot product、并行 SAXPY”这三个点。
- 第 2 节的 Dietrich 方法，主要用于说明相关工作和为什么作者选择 WY。

价值判断：

- 对数值线性代数理解价值：高。
- 对现代高性能 QR / LAPACK 理解价值：高。
- 对 AI 推理系统直接价值：中。它不是 attention 论文，但“把一串低秩/线性更新压成块矩阵乘法”的思想和现代 chunkwise linear attention、WY / associative scan 类推导有相通处。
- 对 kernel / 算子优化价值：中到高。它强调的不是少算，而是把计算组织成硬件友好的密集块操作。
- 是否值得复现：中。如果目标是理解 QR，可以复现；如果目标是现代性能，应直接对照 LAPACK blocked QR 或 GPU QR 实现。

## 11. 下一步阅读建议

如果只是为了理解本文，建议按这个顺序读：

1. 第 3 节，重点看 $Q_k=I+W_kY_k^\top$ 的递推。
2. 第 4 节，重点看 Algorithm 4.2，理解 panel QR 和 trailing update 的分工。
3. 附录，确认 WY 更新为什么仍然稳定。

如果是为了联系现代实现，下一步应该读 LAPACK blocked QR 里的 compact WY / block reflector 表示，特别是形如：

$$
H = I - VTV^\top
$$

的 block reflector，其中 $T$ 是小三角矩阵。它和本文的 $I+WY^\top$ 是同一个设计方向：把多个 Householder 反射打包，然后通过 GEMM 一次性应用。

如果是为了联系线性 attention / DeltaNet 里的 WY 推导，可以重点比较两件事：

- 本文的 WY：压缩一串 Householder 正交变换。
- DeltaNet / Gated DeltaNet 的 WY/UT：压缩一串 rank-1 状态转移。

它们的共同点是：把原本 sequential 的一串矩阵变换，转成可以 chunkwise 并行的矩阵乘法。
