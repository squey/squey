#!/bin/bash
flatpak update --user -y

./subs_vars.sh $@

source flatpak-dev-cli/setup-flatpak-dev.sh .. inendi-inspector.json
