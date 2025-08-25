<div align="center"><h1> EgoMusic-driven Human Dance Motion Estimation with Skeleton Mamba<br>
</h1>
<p align="center">
    <a href="https://scholar.google.com/citations?user=F5Fr2ysAAAAJ&hl=vi" style="text-decoration: none;">Quang Nguyen</a> •
    <a href="https://minhnhatvt.github.io/" style="text-decoration: none;">Nhat Le</a> •
    <a href="https://scholar.google.com/citations?user=unbPvWAAAAAJ&hl=zh-CN" style="text-decoration: none;">Baoru Huang</a> •
    <a href="https://scholar.google.com/citations?hl=th&user=qyExc4QAAAAJ&view_op=list_works" style="text-decoration: none;">Minh Nhat Vu •
    <a href="https://scholar.google.com/citations?user=WbG27wQAAAAJ&hl=en" style="text-decoration: none;">Chengcheng Tang •
    <a href="https://sites.google.com/view/vannguyen" style="text-decoration: none;">Van Nguyen •
    <a href="https://scholar.google.com/citations?user=8ck0k_UAAAAJ&hl=en" style="text-decoration: none;">Ngan Le •
    <a href="https://sites.google.com/tdtu.edu.vn/vongocthieu" style="text-decoration: none;">Thieu Vo</a> •
    <a href="https://www.csc.liv.ac.uk/~anguyen/" style="text-decoration: none;">Anh Nguyen</a>
</p>
<h1><sub><sup><a href="https://iccv.thecvf.com/">ICCV 2025</a></sup></sub></h1>

[![Website](https://img.shields.io/badge/Website-Demo-fedcba?style=flat-square)](https://zquang2202.github.io/SkeletonMamba/) 
[![arXiv](https://img.shields.io/badge/arXiv-2403.07487-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2508.10522)

</div>

# Environment Preparation
Follow these steps to install the GraspMAS framework:

1. **Clone repo:**
    ```bash
    git clone https://github.com/Fsoft-AIC/SkeletonMamba.git
    cd SkeletonMamba
    ```
2. **Prepare environment:**
cuda==11.8,python==3.11, torch==2.2.0, gcc==11.3 (for State Space Model enviroment). Installing Mamba may cost a lot of effort. If you encounter problems, this [issues in Mamba](https://github.com/state-spaces/mamba/issues) may be very helpful.

Install virtual environment
```bash
bash env.sh
```
Install ```mujoco```. 
```
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```
Install PyTorch3D. 
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```

3. **Quick start:**
```python
model = SkeletonMamba(
        input_dim=9,
        embed_dim=128,
        depth=8,
        num_joints=24,
        has_text=True,
        d_context=128,
        device="cuda",
        use_pe=2,
        video_frames = 120,
    ).to("cuda")
x = torch.rand(10, 2880, 9).to("cuda")
t = torch.rand(10).to("cuda")
_context = torch.rand(10, 10, 128).to("cuda")
o = model(x, t, y=_context)
_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Param count: {_param_count}")
print(o.shape)
print(model)
```

# Dataset preparation
The dataset will be published soon!!

# Training and Eval
1. To train the model, run:
```bash
bash scripts/train.sh
```

2. To evaluate, run:
```bash
bash scripts/eval.sh
```

# Citation

Please cite our paper:

```bibtex
@InProceedings{nguyen2025egomusic,
      title={EgoMusic-driven Human Dance Motion Estimation with Skeleton Mamba},
      author={{Nguyen, Quang and Le, Nhat and Huang, Baoru and Vu, Minh Nhat and Tang, Chengcheng and Nguyen, Van and Le, Ngan and Vo, Thieu and Nguyen, Anh},
      booktitle = {ICCV},
      year={2025}
}
```

# Acknowledgement
The code base is develop and adapt from [Zigma](https://github.com/cvlab-columbia/viper), [EgoEgo](https://github.com/duality-robotics/viper/tree/main)