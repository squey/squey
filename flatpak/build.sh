#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source ./subs_vars.sh $@

if [ $EXPORT_BUILD = true ]; then
    flatpak-builder $DIR/build $DIR/inendi-inspector.json --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir=$DIR/gnupg --ccache  --force-clean --delete-build-dirs --repo=$REPO_DIR || exit 1
    flatpak update --user com.esi_inendi.Inspector || exit 2
    flatpak-builder $DIR/build $DIR/pcap-inspector.json --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir=$DIR/gnupg --ccache  --force-clean --delete-build-dirs --repo=$REPO_DIR || exit 3
else
    flatpak-builder $DIR/build $DIR/inendi-inspector.json --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir=$DIR/gnupg --ccache  --force-clean --delete-build-dirs --build-only || exit 4
fi


