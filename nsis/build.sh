#!/bin/bash

makensis -DDISPLAY_NAME="INENDI Inspector" -DPRODUCT_NAME="inendi-inspector" -DFLATPAK_PACKAGE_NAME="org.inendi.Inspector" -DFLATPAKREF_URL="https://inendi.gitlab.io/inspector/flatpak/inspector.flatpakref" inspector.nsi
