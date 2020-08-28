#!/bin/bash 

DOCKER_VERSION=`docker version --format '{{.Server.Version}}'`
NVIDIA_DOCKER_RUNTIME=""

vercomp () {
    if [[ $1 == $2 ]]
    then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++))
    do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++))
    do
        if [[ -z ${ver2[i]} ]]
        then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]}))
        then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]}))
        then
            return 2
        fi
    done
    return 0
}

vercomp ${DOCKER_VERSION} "19.03"
if [[ $? -eq 2 ]]
then
    command -v "nvidia-docker" &> /dev/null && NVIDIA_DOCKER_RUNTIME="--runtime=nvidia" || {
        echo -n "nvidia-docker is required to enable GPU acceleration under Docker < 19.03"
        echo "(visit https://github.com/NVIDIA/nvidia-docker for more information)"
        
        read -p "Continue without GPU acceleration? [y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]
        then
            exit 1
        fi
    }
fi

