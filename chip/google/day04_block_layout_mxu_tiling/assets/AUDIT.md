# SVG audit

Round 1 factual: checked BlockSpec/MXU tiling against JAX Pallas matmul/design docs.
Round 2 semantic: no unpublished VMEM bank/floorplan claimed.
Round 3 rendering: SVG bounds, labels and arrows checked.

The official vector-layout example is stored as `02_vector_layout_example_official.svg` and attributed in `REMOTE_IMAGES.md`; its `8×128` rectangles are layout footprints with padding/overlap, not a claim that six tiles exactly partition the `12×320` logical array. The surrounding three-layer diagram remains explicitly marked as a teaching schematic.
