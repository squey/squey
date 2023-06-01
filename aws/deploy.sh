#!/bin/bash 

INSTALL_DIR="/opt/squey/squey"
source "${INSTALL_DIR}/env.conf"

# Install NVIDIA drivers
apt-get update
apt-get -y install nvidia-driver-430
modprobe nvidia

# Install nvidia-docker2
apt-get install -y docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
systemctl restart docker

# Build Docker image
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"
wget https://squey.gitlab.io/squey/squey_docker.zip
unzip squey_docker.zip
mv env_aws.conf env.conf
./build.sh

# Start container at startup
apt-get -y install systemd-docker
ln -s "${INSTALL_DIR}/squey_docker.service" /lib/systemd/system
systemctl enable squey_docker
systemctl start squey_docker

# Cleanup
apt-get -y autoremove
apt-get -y clean
rm -rf /var/lib/apt/lists/* /var/tmp/*

# Send AWS wait handle
/opt/aws/bin/cfn-signal -e 0 "${wait_handle}"
