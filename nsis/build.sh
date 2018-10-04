#!/bin/bash

makensis -DDISPLAY_NAME="INENDI Inspector" -DPRODUCT_NAME="inendi-inspector" -DFLATPAK_PACKAGE_NAME="com.esi_inendi.Inspector" -DFLATPAKREF_URL="https://repo.esi-inendi.com/inendi-inspector.flatpakref" inspector.nsi
makensis -DDISPLAY_NAME="PCAP Inspector" -DPRODUCT_NAME="pcap-inspector" -DFLATPAK_PACKAGE_NAME="com.pcap_inspector.Inspector" -DFLATPAKREF_URL="https://pcap-inspector.com/inspector.flatpakref" inspector.nsi
