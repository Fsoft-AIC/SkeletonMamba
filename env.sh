conda create -n skm python=3.11
conda activate skm
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch torchvision  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install  torchdiffeq  matplotlib h5py timm diffusers accelerate loguru blobfile ml_collections wandb
pip install hydra-core opencv-python torch-fidelity webdataset einops pytorch_lightning
pip install torchmetrics --upgrade
pip install opencv-python causal-conv1d
cd dis_causal_conv1d && pip install -e . && cd ..
cd dis_mamba && pip install -e . && cd ..
pip install  scikit-learn --upgrade 
pip install transformers==4.36.2
pip install av