#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source ./subs_vars.sh $@

if [ $EXPORT_BUILD = true ]; then
    flatpak-builder $DIR/build $DIR/inendi-inspector.json --state-dir="$DIR/$STATE_DIR" --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir=$DIR/gnupg --ccache --force-clean --delete-build-dirs --repo=$REPO_DIR || exit 1
    flatpak remote-add --user --if-not-exists --no-gpg-verify `basename "$REPO_DIR"` "$REPO_DIR" || exit 2
    flatpak install --user -y `basename "$REPO_DIR"` com.esi_inendi.Inspector/x86_64/$BRANCH_NAME
    flatpak update --user -y com.esi_inendi.Inspector/x86_64/$BRANCH_NAME || exit 3
    if [ $EXPORT_PCAP_BUILD = true ]; then
        flatpak-builder $DIR/build $DIR/pcap-inspector.json --state-dir="$DIR/$STATE_DIR" --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir=$DIR/gnupg --ccache --force-clean --delete-build-dirs --repo=$REPO_DIR || exit 4
    fi
else
    flatpak-builder $DIR/build $DIR/inendi-inspector.json --state-dir="$DIR/$STATE_DIR" --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir=$DIR/gnupg --ccache --force-clean --delete-build-dirs --build-only || exit 5
fi

if [ ! -z "$UPLOAD_URL" -a ! -z "$REPO_DIR" ]; then
    ./ostree-releng-scripts/rsync-repos --rsync-opt "-e ssh -p $UPLOAD_PORT" --src $REPO_DIR/ --dest $UPLOAD_URL
fi
