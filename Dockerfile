FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# 安裝基本工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-dev \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 升級 pip
RUN pip install --upgrade pip

# 安裝 DARTS 所需套件
RUN pip install torchvision \
    numpy \
    matplotlib \
    scipy \
    wandb

# 建立工作目錄
WORKDIR /workspace

# 預設命令：進入 bash
CMD ["/bin/bash"]
