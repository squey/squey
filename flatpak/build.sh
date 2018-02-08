#!/bin/bash
flatpak-builder build manifest.json --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir=./gnupg --ccache  --force-clean --repo=repo
