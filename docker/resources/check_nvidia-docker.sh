#!/bin/bash 

NVIDIA_DOCKER_RUNTIME=""
command -v "nvidia-docker" &> /dev/null && NVIDIA_DOCKER_RUNTIME="--runtime=nvidia" || {
    echo -n "nvidia-docker is required to enable GPU acceleration "
    echo "(visit https://github.com/NVIDIA/nvidia-docker for more information)"
    
    read -p "Continue without GPU acceleration? [y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
}

