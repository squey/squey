#!/bin/bash

find "${INSTALL_PATH}" -name "nice-dcv*.tgz" -exec tar zxvf "{}" --no-same-owner -C "/opt" \;
rm -rf "${INSTALL_PATH}"/nice-dcv*.tgz

cd /opt/nice-dcv-*/
DCV_PACKAGES_DIR=$(pwd)
cd -
TMP_DCV_SERVER_DIR=/tmp/nice-dcv-server
mkdir -p "$TMP_DCV_SERVER_DIR"

ar x "$DCV_PACKAGES_DIR"/nice-dcv-server_*.deb --output "$TMP_DCV_SERVER_DIR" &> /dev/null
rm -rf "$DCV_PACKAGES_DIR"/nice-dcv-server_*.deb
mkdir -p "$TMP_DCV_SERVER_DIR"/control
tar Jxvf "$TMP_DCV_SERVER_DIR"/control.tar.xz -C "$TMP_DCV_SERVER_DIR"/control &> /dev/null
sed '13d' -i "$TMP_DCV_SERVER_DIR"/control/postinst
cd "$TMP_DCV_SERVER_DIR/control"
tar Jcvf "$TMP_DCV_SERVER_DIR/control.tar.xz" . &> /dev/null
cd -
rm -rf "$TMP_DCV_SERVER_DIR"/control
ar -r "$DCV_PACKAGES_DIR"/nice-dcv-server.deb "$TMP_DCV_SERVER_DIR/debian-binary" "$TMP_DCV_SERVER_DIR/control.tar.xz" "$TMP_DCV_SERVER_DIR/data.tar.xz"
rm -rf "$DCV_PACKAGES_DIR"/nice-dcv-gl*.deb
dpkg -i "$DCV_PACKAGES_DIR"/*.deb || true
rm -rf "$TMP_DCV_SERVER_DIR"
apt-get --fix-broken install -y
sed -i '6i ExecStartPre=-dcv _idl -n' /lib/systemd/system/dcvserver.service
rm -rf "$DCV_PACKAGES_DIR"