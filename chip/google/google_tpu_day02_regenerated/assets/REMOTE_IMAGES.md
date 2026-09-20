# 本地资料图记录

正文中的两张资料图已下载到当前 `assets/` 目录，正文只引用本地文件。原正文使用的 `.../assets/img/tpu.png` 已失效；JAX Scaling Book 当前对应文件是 `tpu-chip.png`。

| 文件 | 用途 | 来源页面 | 原始文件 | 抓取日期 | 尺寸 / 校验 |
|---|---|---|---|---|---|
| `01_tpu_chip_jax_scaling_book.png` | TensorCore、HBM、VMEM、MXU/VPU/Scalar Unit 的抽象关系 | [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) | [Scaling Book commit `ac72f02`](https://github.com/jax-ml/scaling-book/blob/ac72f020e4320b02e54b6f1b2324cdf1c579388c/assets/img/tpu-chip.png) | 2026-09-08 | 1600×802 PNG; SHA-256 `3c8878d42e9b43a0c22517f36a30bd14d1810044fa4dd0c83cb71a289d27f6dc` |
| `02_tpu_systolic_array_google.png` | 第一代 TPU systolic-array 数据流示意 | [An in-depth look at Google’s first TPU](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu) | [Google Cloud Blog static image](https://storage.googleapis.com/gweb-cloudblog-publish/images/tpu-17u39j.max-500x500.PNG) | 2026-09-08 | 500×412 PNG; SHA-256 `bf34d59b15faf19fbab6fbde22bbd5aba3a6ec3be8ba0b9ead0ebc6592c8084b` |

## 使用与许可说明

- JAX Scaling Book 仓库采用 [MIT License](https://github.com/jax-ml/scaling-book/blob/ac72f020e4320b02e54b6f1b2324cdf1c579388c/LICENSE)；保留项目、来源页面和许可证链接。
- Google Cloud Blog 没有对该嵌图单独声明开源许可证。这里保存未裁剪、未改色的本地副本用于带来源的学习引用，不把它标成可自由再分发素材。
- 两张图都是架构/数据流示意：前者不是 Ironwood floorplan，后者也不能用来推断 Ironwood 的 MXU 尺寸或版图。
