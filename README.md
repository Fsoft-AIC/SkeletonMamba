<div align="center">
<h1>EgoMusic-driven Human Dance Motion Estimation with Skeleton Mamba</h1>

<p>
    <a href="https://scholar.google.com/citations?user=F5Fr2ysAAAAJ&hl=vi">Quang Nguyen</a> •
    <a href="https://minhnhatvt.github.io/">Nhat Le</a> •
    <a href="https://scholar.google.com/citations?user=unbPvWAAAAAJ&hl=zh-CN">Baoru Huang</a> •
    <a href="https://scholar.google.com/citations?hl=th&user=qyExc4QAAAAJ&view_op=list_works">Minh Nhat Vu</a> •
    <a href="https://scholar.google.com/citations?user=WbG27wQAAAAJ&hl=en">Chengcheng Tang</a> •
    <a href="https://sites.google.com/view/vannguyen">Van Nguyen</a> •
    <a href="https://scholar.google.com/citations?user=8ck0k_UAAAAJ&hl=en">Ngan Le</a> •
    <a href="https://sites.google.com/tdtu.edu.vn/vongocthieu">Thieu Vo</a> •
    <a href="https://www.csc.liv.ac.uk/~anguyen/">Anh Nguyen</a>
</p>

<p><strong>ICCV 2025</strong></p>

[![Website](https://img.shields.io/badge/Website-Demo-fedcba?style=flat-square)](https://zquang2202.github.io/SkeletonMamba/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.10522-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2508.10522)

</div>

## Installation

```bash
SKELETONMAMBA_PYTHON=python3.12 bash env.sh
source .venv-cuda/bin/activate
python -m scripts.check_cuda
```

## Data preparation
Video preparation requires system `ffprobe`. SMPL geometry export requires the optional `smplx` package and a licensed SMPL model matching the dataset. Export its geometry, prepare each split, and compute training statistics:

```bash
python -m scripts.export_smpl_geometry --model-path /path/to/SMPL_MALE.pkl \
  --gender male --betas 0 0 0 0 0 0 0 0 0 0 --up-axis z \
  --output data/processed/geometry.npz
for split in train val test; do
  python -m scripts.prepare_motion --manifest "data/processed/${split}.jsonl" \
    --geometry data/processed/geometry.npz
done
python -m scripts.compute_statistics --manifest data/processed/train.jsonl \
  --output data/processed/statistics.npz
```

The model uses 4800-dimensional Jukebox audio features and 2048-dimensional ResNet50 video features. Audio offsets must be verified against the source motion. Follow [audio feature preparation](docs/AUDIO_AND_BEATS.md) to import the supplied caches or extract features. For timestamped source audio caches and raw video:

```bash
python -m scripts.prepare_audio --manifest data/processed/train.jsonl \
  --source-dir /path/to/source_jukebox_npz --offsets /path/to/verified_offsets.json \
  --output-dir data/processed/audio --output-manifest data/processed/train_audio.jsonl
python -m scripts.prepare_video --manifest data/processed/train_audio.jsonl \
  --extract --device cuda --weights /path/to/resnet50_state_dict.pt \
  --output-dir data/processed/video --output-manifest data/processed/train.prepared.jsonl
```

## Training and inference

Set the data paths in [configs/egoaistpp_cuda.yaml](configs/egoaistpp_cuda.yaml), then train or resume:

```bash
python -m train --config configs/egoaistpp_cuda.yaml
python -m train --config configs/egoaistpp_cuda.yaml \
  --resume Experiments/cuda/latest.pt
```

Checkpoints include model and EMA weights, optimizer state, normalization, geometry, and the configuration. Resume requires matching training settings and data. Use a separate output directory for each experiment.

Generate motion from a prepared audio/video pair:

```bash
python -m infer --checkpoint Experiments/cuda/best.pt \
  --audio /path/to/audio.npz --video /path/to/video.npz \
  --output Experiments/cuda/prediction.npz --device cuda
```

## Evaluation

```bash
python -m eval --checkpoint Experiments/cuda/best.pt \
  --manifest data/processed/test.prepared.jsonl \
  --output-dir Experiments/cuda/test --device cuda
```

## Citation

```bibtex
@InProceedings{nguyen2025egomusic,
  title = {EgoMusic-driven Human Dance Motion Estimation with Skeleton Mamba},
  author = {Nguyen, Quang and Le, Nhat and Huang, Baoru and Vu, Minh Nhat and Tang, Chengcheng and Nguyen, Van and Le, Ngan and Vo, Thieu and Nguyen, Anh},
  booktitle = {ICCV},
  year = {2025}
}
```

## Acknowledgements

The implementation builds on [Mamba/SSD](https://github.com/state-spaces/mamba), [ZigMa](https://github.com/CompVis/zigma), [EDGE](https://github.com/Stanford-TML/EDGE), and [EgoEgo](https://github.com/lijiaman/egoego_release). Preserve upstream notices and the licenses associated with code, datasets, and model assets.
