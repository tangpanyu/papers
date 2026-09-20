# SVG audit

Round 1 factual: checked against JAX Pallas TPU Pipelining/Quickstart.
Round 2 semantic: A/B are logical VMEM slots; no physical topology claimed.
Round 3 rendering: labels and geometry checked for overlap/bounds.

Official references are stored locally and tracked in `REMOTE_IMAGES.md`: JAX's TPU memory-space cartoon and bandwidth-bound software-pipeline timeline. The official figures are kept distinct from the self-drawn double-buffer schematic.

The pinned bandwidth-bound SVG contains several upstream-exported path sets outside the visible canvas; the in-canvas set around `y≈30–85` is the intended `copy_in`/`compute` timeline, while other sets are clipped export remnants. The local file is byte-identical to the pinned commit, JAX `main`, and the current docs asset; rendering with its original `viewBox`/`clipPath` produces the complete 1100×140 `copy_in`/`compute`/`Idle` timeline. Do not widen the `viewBox` to expose those clipped remnants.
