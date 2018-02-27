#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
flatpak-builder $DIR/build $DIR/manifest.json --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir=$DIR/gnupg --ccache  --force-clean --repo=$DIR/repo || exit 1
flatpak update --user com.esi_inendi.Inspector || exit 2
flatpak-builder $DIR/build $DIR/pcap-inspector.json --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir=$DIR/gnupg --ccache  --force-clean --repo=$DIR/repo || exit 3
