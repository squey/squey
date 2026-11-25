#!/bin/bash
#
# Installs into a prefix what the desktop learns of Squey from the Flatpak: the
# icon, the desktop entry, the MIME types it opens, and the metainfo
# generate_appstream_metadata.sh writes. "flatpak build-finish" exports them out
# of the share/ of the prefix. org.squey.Squey_export.bst runs it.
#
# Usage: install_desktop_integration.sh <prefix>

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <prefix>" >&2
  exit 1
fi
PREFIX="$1"

FILES_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck source=buildstream/files/flatpak/metadata.sh
source "$FILES_DIR/flatpak/metadata.sh"
GENERATED_DIR="$(mktemp -d)"
trap 'rm -rf "$GENERATED_DIR"' EXIT

install -D -m 0644 "$FILES_DIR/org.squey.Squey.png" -t "$PREFIX/share/icons/hicolor/128x128/apps/"
install -D -m 0644 "$FILES_DIR/org.squey.Squey.png" -t "$PREFIX/share/app-info/icons/flatpak/128x128/"

cat <<EOF >"$GENERATED_DIR/org.squey.Squey.desktop"
[Desktop Entry]
Type=Application
Name=Squey
Exec=$FLATPAK_COMMAND %F
Icon=org.squey.Squey
Categories=Science;Graphics;Network;
MimeType=application/org.squey.investigation;text/csv;text/tab-separated-values;application/vnd.apache.parquet;application/vnd.tcpdump.pcap;application/x-pcapng;
EOF
install -D -m 0644 "$GENERATED_DIR/org.squey.Squey.desktop" -t "$PREFIX/share/applications"

# What the desktop entry above names has to exist in the shared MIME
# database, or the association is written against a type no file is ever
# given. Declared here are the types that database does not carry -- the
# investigation is ours, and Parquet and pcapng reached it late enough that
# a supported distribution may still be without them -- and the compressed
# spellings it otherwise resolves to their container: *.csv.gz is longer
# than *.gz, so the glob below wins and the file is offered to Squey.
#
# A type that is already known keeps its own comment: re-stating it here
# would replace a translated description with an English one.
cat <<EOF >"$GENERATED_DIR/org.squey.Squey.xml"
<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
    <mime-type type="application/org.squey.investigation">
        <comment>Squey investigation</comment>
        <glob pattern="*.pvi"/>
    </mime-type>
    <mime-type type="application/vnd.apache.parquet">
        <comment>Apache Parquet file</comment>
        <glob pattern="*.parquet"/>
    </mime-type>
    <mime-type type="application/x-pcapng">
        <comment>PCAPNG capture file</comment>
        <glob pattern="*.pcapng"/>
    </mime-type>
    <mime-type type="text/csv">
        <glob pattern="*.csv.zip"/>
        <glob pattern="*.csv.gz"/>
        <glob pattern="*.csv.bz2"/>
        <glob pattern="*.csv.xz"/>
        <glob pattern="*.csv.zst"/>
    </mime-type>
    <mime-type type="text/tab-separated-values">
        <glob pattern="*.tsv.zip"/>
        <glob pattern="*.tsv.gz"/>
        <glob pattern="*.tsv.bz2"/>
        <glob pattern="*.tsv.xz"/>
        <glob pattern="*.tsv.zst"/>
    </mime-type>
</mime-info>
EOF
install -D -m 0644 "$GENERATED_DIR/org.squey.Squey.xml" -t "$PREFIX/share/mime/packages"

install -D -m 0644 "$FILES_DIR/org.squey.Squey.metainfo.xml" -t "$PREFIX/share/metainfo"
