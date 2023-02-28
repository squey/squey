#!/bin/bash

: "${DOCKER:=buildah}"

command -v "${DOCKER}" &> /dev/null || { echo >&2 "'${DOCKER}' executable is required to execute this script."; exit 1; }

# Source environment variables
source env.conf
source resources/.env.conf
source resources/check_nvidia-docker.sh

# Copy SSL certificates
if [[ -f "${DCV_SSL_KEY_PATH}" ]] && [[ -f "${DCV_SSL_CERT_PATH}" ]]
then
    cp "${DCV_SSL_CERT_PATH}" "${DCV_SSL_KEY_PATH}" resources
fi

# Configure installation type (online/offline)
mkdir -p "data"
INSTALL_MODE="online"
[ "$(ls -A data)" ] && INSTALL_MODE="offline"
echo "INSTALL_MODE=$INSTALL_MODE"

OPTS=$([ -z "$PRODUCTION" ] && echo "--layers" || echo "--squash")
${DOCKER} build-using-dockerfile $OPTS \
    --build-arg INSTALL_PATH="${INSTALL_PATH}" \
    --build-arg INSTALL_MODE="${INSTALL_MODE}" \
    --build-arg DCV_LICENSE_SERVER="${DCV_LICENSE_SERVER}" \
    --build-arg APT_PROXY="${APT_PROXY}" \
    -f resources/Dockerfile --tag inendi/inspector
