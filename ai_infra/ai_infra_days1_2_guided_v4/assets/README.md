Markdown 直接引用以下官方原图 URL，以保证使用原图而非自绘替代：

- NVIDIA PTX Figure 182:
  https://docs.nvidia.com/cuda/parallel-thread-execution/_images/tensor-memory-layout.png
- NVIDIA PTX Figure 183:
  https://docs.nvidia.com/cuda/parallel-thread-execution/_images/tcgen05-mma-fragment-3232b.png
- vLLM Figure 7:
  https://vllm.ai/blog-assets/figures/2025-vllm-anatomy/prefix_pt2.png
- vLLM Figure 8:
  https://vllm.ai/blog-assets/figures/2025-vllm-anatomy/prefix_pt3.png
- vLLM Figure 4:
  https://vllm.ai/blog-assets/figures/2025-vllm-anatomy/fwd_pass.png

当前执行容器无外网 DNS，因此无法把原图 bytes 物理落到 `assets/`；Typora 联网打开 Markdown 时会直接拉取官方图片。