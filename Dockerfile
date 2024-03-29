


#Docker file for HuggingFace. To build/run locally see Dockerfile-local




# Python version 3.9.18
#FROM python@sha256:17d96c91156bd5941ca1b6f70606254f7f98be8dbe662c34f41a0080fd490b0c
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

RUN mkdir -m 777 /tmp/NUMBA_CACHE_DIR /tmp/MPLCONFIGDIR
ENV NUMBA_CACHE_DIR=/tmp/NUMBA_CACHE_DIR/
ENV MPLCONFIGDIR=/tmp/MPLCONFIGDIR/
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get install -y \
    build-essential \
    checkinstall \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-distutils \
    curl \
    software-properties-common \
    git \
    libfftw3-dev \
    liblapack-dev \
    libsndfile-dev \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && python3.9 -m pip install --upgrade pip \
    && python3.9 -m pip install cython==0.29.19 \
    && python3.9 -m pip install tifresi==0.1.2 \
    && python3.9 -m pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt ./
RUN python3.9 -m pip install -r requirements.txt

COPY . ./

RUN mkdir -p checkpoints/stylegan2/greatesthits \
    checkpoints/stylegan2/dcase \
    checkpoints/encoder/greatesthits \
    checkpoints/encoder/dcase \
    && wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/audio-stylegan2/greatesthits/network-snapshot-002800.pkl -P checkpoints/stylegan2/greatesthits \
    && wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/audio-stylegan2/dcase/network-snapshot-002200.pkl -P checkpoints/stylegan2/dcase \
    && wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/encoder/greatesthits/netE_epoch_best.pth -P checkpoints/encoder/greatesthits \
    && wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/encoder/dcase/netE_epoch_best.pth -P checkpoints/encoder/dcase

EXPOSE 7860

WORKDIR /app/interface

COPY ./script.sh /
RUN chmod +x /script.sh
ENTRYPOINT ["/script.sh"]