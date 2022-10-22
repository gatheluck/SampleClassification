# Sample Classification

![python versions](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)
[![tests](https://github.com/cvpaperchallenge/Ascender/actions/workflows/lint-and-test.yaml/badge.svg)](https://github.com/cvpaperchallenge/Ascender/actions/workflows/lint-and-test.yaml)
[![MIT License](https://img.shields.io/github/license/cvpaperchallenge/Ascender?color=green)](LICENSE)

## What is this repository?

This is a sample repository for image classification application.
It is primarily intended to be used by other repositories as an example of an ML application when explaining how to deploy an ML application.
As an example, CIFAR-10 image classifier is used.

## Project Organization

```
    ├── .github/           <- Settings for GitHub.
    │
    ├── environments/      <- Provision depends on environments.
    │
    ├── models/            <- Pretrained and serialized models.
    │
    ├── src/               <- Source code. This sould be Python module.
    │   │
    │   ├── lambda         <- Codes used when deployed as AWS Lambda.
    │   │
    │   ├── ml             <- Core ML logic.
    │   │
    │   └── script         <- Some scripts for entrypoint. 
    │
    ├── tests/             <- Test codes.
    │
    ├── .flake8            <- Setting file for Flake8.
    ├── .dockerignore
    ├── .gitignore
    ├── LICENSE
    ├── Makefile           <- Makefile used as task runner. 
    ├── poetry.lock        <- Lock file. DON'T edit this file manually.
    ├── poetry.toml        <- Setting file for Poetry.
    ├── pyproject.toml     <- Setting file for Project. (Poetry, Black, isort, Mypy)
    └── README.md          <- The top-level README for developers.

```

## Prerequisites

- [Docker](https://www.docker.com/) 
- [Docker Compose](https://github.com/docker/compose)
- (Optional) [NVIDIA Container Toolkit (nvidia-docker2)](https://github.com/NVIDIA/nvidia-docker)

**NOTE**: Example codes in the README.md are written for `Docker Compose v2`. However, Ascender also should work under `Docker Compose v1`. If you are using `Docker Compose v1`, just replace `docker compose` in the example code by `docker-compose`.

## Prerequisites installation

Here, we show example prerequisites installation codes for Ubuntu. If prerequisites  are already installed your environment, please skip this section. If you want to install in another environment, please follow the officail documentations.

- Docker and Docker Compose: [Install Docker Engine](https://docs.docker.com/engine/install/)
- NVIDIA Container Toolkit (nvidia-docker2): [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)


### Install Docker and Docker Compose

```bash
# Set up the repository
$ sudo apt update
$ sudo apt install ca-certificates curl gnupg lsb-release
$ sudo mkdir -p /etc/apt/keyrings
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
$ echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker and Docker Compose
$ sudo apt update
$ sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

If `sudo docker run hello-world` works, installation succeeded.

### (Optional) NVIDIA Container Toolkit

If you want to use GPU in Ascender, please install NVIDIA Container Toolkit (nvidia-docker2) too. NVIDIA Container Toolkit also requires some prerequisites to install. So please check thier [official documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#pre-requisites) first.

```bash
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

$ sudo apt update
$ sudo apt install -y nvidia-docker2
$ sudo systemctl restart docker
```

If `sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi` works, installation succeeded.

## Quick start

Here we explain how to run sample classification.

First, please visit following link and download a checkpoint from google drive. Downloaded checkpoint should be placed under `models/`.
https://drive.google.com/drive/folders/1JlHNlrR4e-3NKCJh85lWOOqNs2fLodN0?usp=sharing

After that, you can run classification task by following commands:

```bash
# Build Docker image and run container
% cd environments/cpu
% sudo docker compose up -d

# Run bash inside of container (jump into contaienr)
$ sudo docker compose exec core bash

# Create virtual environment and install dependent packages by Poetry
% poetry install

# Run classification
% poetry run python3 src/script/predict.py

{'probabilities': [{'airplane': 0.4703303575515747, 'ship': 0.4138832986354828, 'truck': 0.04285023361444473}]}

# <Exit from container by ctrl+c>

# Stop and remove container
$ sudo docker compose down
```
