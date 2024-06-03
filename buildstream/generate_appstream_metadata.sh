#!/bin/bash

set -e
set -x

# Generate AppStream metadata
CHANGELOG_CONTENT="$(awk '/^---.*---$/ {count++} count == 1 {print} count == 2 {exit}' ../CHANGELOG | head -n -2 | tail -n +3)"
CHANGELOG_CONTENT="$(echo "$CHANGELOG_CONTENT" | sed '1 s|\(.*\)|<p>\1</p>\n          <ul>|' | sed 's|\* \(.*\)|            <li>\1</li>|' | (tee -a && echo '          </ul>'))"
PROJECT_DESCRIPTION="$(sed -n '/<!-- project_description_start -->/,/<!-- project_description_end -->/p' ../README.md | sed -e 's/<\/\?\(b\|a[^>]*\|br\)>//g' | sed '1d;$d')"
jinja2 -D project_description="$PROJECT_DESCRIPTION" -D version="$(cat ../VERSION.txt | tr -d '\n')" -D date="$(date --iso)" -D changelog="$CHANGELOG_CONTENT" files/org.squey.Squey.metainfo.xml.j2 > files/org.squey.Squey.metainfo.xml

if [ "$APPSTREAM_LINTER" = true ]; then
  flatpak-builder-lint appstream files/org.squey.Squey.metainfo.xml
fi