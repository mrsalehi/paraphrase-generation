#!/usr/bin/env bash

CUDA_REPO_PKG=cuda-repo-ubuntu1604_10.0.130-1_amd64.deb

wget -O /tmp/${CUDA_REPO_PKG} http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}

sudo dpkg -i /tmp/${CUDA_REPO_PKG}

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

rm -f /tmp/${CUDA_REPO_PKG}

sudo apt-get update

sudo apt-get install -y cuda-drivers