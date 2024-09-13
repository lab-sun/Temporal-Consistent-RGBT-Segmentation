FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# 更新密钥
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# 更新和安装必要的软件包
RUN apt-get update && apt-get install -y \
    vim \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# 升级 pip 和 setuptools
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=40.3.0 

# 安装 Python 包
RUN pip3 install -U scipy scikit-learn
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchsummary
RUN pip3 install tensorboard==2.11.0
RUN pip3 install einops
RUN pip3 install easydict
RUN pip3 install six
RUN pip3 install timm
RUN pip3 uninstall -y Pillow
RUN pip3 install Pillow==9.5.0

# 安装 matplotlib 和 opencv
RUN pip3 install matplotlib
RUN pip3 install opencv-python-headless
