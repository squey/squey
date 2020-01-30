#!/bin/bash

source env.conf
source resources/.env.conf
source resources/check_nvidia-docker.sh

INSTALL_MODE="online"
[[ -d "data" ]] && INSTALL_MODE="offline"

# Build base image
docker build --build-arg INSTALL_PATH="${INSTALL_PATH}" --build-arg INSTALL_MODE="${INSTALL_MODE}" --build-arg DCV_LICENSE_SERVER="${DCV_LICENSE_SERVER}" --build-arg INSPECTOR_LICENSE_SERVER="${INSPECTOR_LICENSE_SERVER}" --build-arg APT_PROXY="${APT_PROXY}" . -f resources/Dockerfile -t inendi/inspector

# Install INENDI Inspector
docker run --privileged ${NVIDIA_DOCKER_RUNTIME} --rm --name inspector-install -v /sys/fs/cgroup:/sys/fs/cgroup:ro -d inendi/inspector
docker cp resources/install_files.sh inspector-install:"${INSTALL_PATH}/"
docker exec inspector-install bash "${INSTALL_PATH}/install_files.sh" "${INSTALL_MODE}"

docker cp resources/configure_auth.sh inspector-install:"${INSTALL_PATH}/"
docker exec inspector-install bash "${INSTALL_PATH}/configure_auth.sh"

# Configure DCV SSL certificate
if [[ -f "${DCV_SSL_KEY_PATH}" ]] && [[ -f "${DCV_SSL_CERT_PATH}" ]]
then
    docker cp "${DCV_SSL_CERT_PATH}" inspector-install:"${INSTALL_PATH}/"
    docker cp "${DCV_SSL_KEY_PATH}" inspector-install:"${INSTALL_PATH}/"

fi
docker cp resources/configure_ssl.sh inspector-install:"${INSTALL_PATH}/"
docker exec inspector-install bash "${INSTALL_PATH}/configure_ssl.sh"

docker commit inspector-install inendi/inspector

function finish {
    docker stop inspector-install
}
trap finish EXIT TERM KILL

chmod +x run.sh
