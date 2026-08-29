# LabelFormer.mojo

[![CI](https://github.com/labelrefinery/LabelFormer.mojo/actions/workflows/ci.yml/badge.svg)](https://github.com/labelrefinery/LabelFormer.mojo/actions/workflows/ci.yml) [![mojoshelf](https://mojoshelf.org/badge/labelformer.svg)](https://mojoshelf.org/tins/labelformer) [![mojo nightly](https://mojoshelf.org/badge/labelformer/nightly.svg)](https://mojoshelf.org/tins/labelformer)

Pure-[Mojo](https://www.modular.com/mojo) inference implementation of **LabelFormer** — *Object Trajectory Refinement for Offboard Perception from LiDAR Point Clouds* (Yang et al., CoRL 2023, [arXiv:2311.01444](https://arxiv.org/abs/2311.01444)).

LabelFormer refines noisy object trajectories (auto-labels) from LiDAR: each frame's box + object points are encoded independently (box MLP + PointPillars-style CNN), a transformer with ALiBi relative position biases reasons over the full trajectory, and the model decodes refined per-frame poses plus a single trajectory-level object size.

The PyTorch reference implementation and training code (ArgoVerse 2) live at [labelrefinery/LabelFormer.py](https://github.com/labelrefinery/LabelFormer.py). This repo reimplements the full forward pass from scratch in Mojo — no MAX, no Python interop at inference time — and verifies numerical parity against PyTorch.

## Layout

- `src/labelformer/` — the package: `tensor` (minimal f32 tensor + matmul), `io` (LFT1 weight/sample container reader), `layers` (layernorm, conv2d, nearest upsample), `pillars` (point featurization + scatter-max pillar grid), `attention` (ALiBi multi-head attention), `model` (full forward; BatchNorms are folded into convs at export).
- `src/main.mojo` — inferencer CLI with per-stage parity reporting.
- `src/debug.mojo` — stage-by-stage divergence localization against a PyTorch debug dump.
- `tests/test_ops.mojo` — hand-computed unit tests for every op.

## Supported Mojo versions

Both **stable Mojo 1.0** and the **Modular nightly** are supported and tested in CI (unit tests + full PyTorch parity, Linux and macOS-arm64):

| pixi environment | compiler | run it |
|---|---|---|
| `default` | nightly (`modular` ≥ 26.6 nightly) | `pixi run test` / `pixi run infer` |
| `stable` | `mojo-compiler == 1.0.0` | `pixi run -e stable test` / `pixi run -e stable infer` |

The mojoshelf tin builds its package with stable 1.0 (the `pixi-build-mojo` backend requires it) and declares `mojo-compiler >=1.0,<2` as its run dependency.

## Install as a mojoshelf tin

Published on [mojoshelf](https://mojoshelf.org/tins/labelformer) as `labelformer`:

```sh
pixi shelf add labelformer     # pixi mode (git source dependency)
shelf add labelformer          # or as a git submodule
```

Maintainers release new versions with `shelf publish` from the repo root
(see [getting started](https://mojoshelf.org/getting-started)).

## Setup

Requires [pixi](https://pixi.sh). `pixi install` pulls the toolchains declared in `pixi.toml`.

Export weights + test samples from the PyTorch side (in a LabelFormer.py checkout with a trained checkpoint):

```sh
uv run python scripts/export_mojo.py --checkpoint runs/smoke/best.pt --out export
cp export/*.lft ../LabelFormer.mojo/data/
```

## Run

```sh
pixi run test    # 14 op unit tests
pixi run infer   # run the exported samples, report per-stage parity vs PyTorch
# or directly:
pixi run mojo run src/main.mojo data/weights.lft data/sample_0.lft
```

## Parity vs PyTorch

Verified on 3 real ArgoVerse 2 val trajectories (T=43–48 frames, 9–3295 lidar points, smoke checkpoint) — max abs difference per stage:

| stage | max \|diff\| |
|---|---|
| tokens (box + pillar CNN fusion) | ≤ 3.1e-6 |
| transformer hidden | ≤ 5.8e-6 |
| pose residuals | ≤ 4.6e-7 |
| size residual | ≤ 2.3e-9 |
| refined boxes | ≤ 1.5e-6 |

One porting subtlety worth knowing: PyTorch computes the pillar-grid readout cell in Python float64, where `(0−(−9.6))/0.2 = 47.999…` truncates to 47; a float32 reproduction gets exactly 48. The exported `__config__` therefore carries the readout cell computed with PyTorch's exact semantics.

## Pretrained weights

Exported weights and parity samples for the smoke checkpoint are on HuggingFace: [mseritan/LabelFormer-AV2-smoke](https://huggingface.co/mseritan/LabelFormer-AV2-smoke) (`mojo/` folder; CC BY-NC-SA, research use) — drop them into `data/` to run without the PyTorch side.
