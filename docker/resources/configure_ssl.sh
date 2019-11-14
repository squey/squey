#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${DIR}/env.conf"
source "${DIR}/.env.conf"

DEST_DCV_SSL_KEY_PATH="/etc/dcv/dcv.key"
DEST_DCV_SSL_CERT_PATH="/etc/dcv/dcv.pem"

default_cert=true
if [[ ! -z "${DCV_SSL_KEY_PATH}" ]] && [[ ! -z "${DCV_SSL_CERT_PATH}" ]]
then
    DCV_SSL_KEY_FILE="$(basename -- ${INSTALL_PATH}/${DCV_SSL_KEY_PATH})"
    DCV_SSL_CERT_FILE="$(basename -- ${INSTALL_PATH}/${DCV_SSL_CERT_PATH})"

    if [[ -f "${INSTALL_PATH}/${DCV_SSL_KEY_FILE}" ]] && [[ -f "${INSTALL_PATH}/${DCV_SSL_CERT_FILE}" ]]
    then
        chown dcv: "${INSTALL_PATH}/${DCV_SSL_KEY_FILE}" "${INSTALL_PATH}/${DCV_SSL_CERT_FILE}"
        chmod 600 "${INSTALL_PATH}/${DCV_SSL_KEY_FILE}" "${INSTALL_PATH}/${DCV_SSL_CERT_FILE}"
        rm -rf "${DEST_DCV_SSL_KEY_PATH}" "${DEST_DCV_SSL_CERT_PATH}" &> /dev/null || true
        mv "${INSTALL_PATH}/${DCV_SSL_KEY_FILE}" "${DEST_DCV_SSL_KEY_PATH}"
        mv "${INSTALL_PATH}/${DCV_SSL_CERT_FILE}" "${DEST_DCV_SSL_CERT_PATH}"
        default_cert=false
    fi
fi

if [ "$default_cert" = true ]
then
    >&2 echo "No SSL certificate provided, using self-signed default one."
    ln -s /var/lib/dcv/.config/NICE/dcv/dcv.key "${DEST_DCV_SSL_KEY_PATH}" &> /dev/null || true
    ln -s /var/lib/dcv/.config/NICE/dcv/dcv.pem "${DEST_DCV_SSL_CERT_PATH}" &> /dev/null || true
else
    echo "Using '${DCV_SSL_CERT_FILE}'/'${DCV_SSL_KEY_FILE}' provided certificates."
fi
