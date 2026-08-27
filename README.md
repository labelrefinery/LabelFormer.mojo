# LabelFormer.mojo

Implementation of **LabelFormer** — *Object Trajectory Refinement for Offboard Perception from LiDAR Point Clouds* (Yang et al., CoRL 2023, [arXiv:2311.01444](https://arxiv.org/abs/2311.01444)).

LabelFormer refines noisy object trajectories (auto-labels) from LiDAR: each frame's box + object points are encoded independently, a transformer with ALiBi relative position biases reasons over the full trajectory, and the model decodes refined per-frame poses plus a single trajectory-level object size.

## Layout

- [`train/`](train/) — PyTorch reference implementation, trained on the [ArgoVerse 2](https://www.argoverse.org/av2.html) sensor dataset. See its README for setup, data download, training, and evaluation.
- Repo root is reserved for a future Mojo port.
