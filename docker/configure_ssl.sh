#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${DIR}/config.env"
source "${DIR}/.config.env"

DCV_SSL_KEY_FILE="$(basename -- ${DCV_SSL_KEY_PATH})"
DCV_SSL_CERT_FILE="$(basename -- ${DCV_SSL_CERT_PATH})"

chown dcv: "${INSTALL_PATH}/${DCV_SSL_KEY_FILE}" "${INSTALL_PATH}/${DCV_SSL_CERT_FILE}"
chmod 600 "${INSTALL_PATH}/${DCV_SSL_KEY_FILE}" "${INSTALL_PATH}/${DCV_SSL_CERT_FILE}"
mv "${INSTALL_PATH}/${DCV_SSL_KEY_FILE}" "/etc/dcv/dcv.key"
mv "${INSTALL_PATH}/${DCV_SSL_CERT_FILE}" "/etc/dcv/dcv.pem"
