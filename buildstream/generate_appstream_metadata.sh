#!/bin/bash

set -e
set -x

# Generate AppStream metadata
CHANGELOG_CONTENT="$(awk '/^---.*---$/ {count++} count == 1 {print} count == 2 {exit}' ../CHANGELOG | head -n -2 | tail -n +3)"
CHANGELOG_CONTENT="$(echo "$CHANGELOG_CONTENT" | sed '1 s|\(.*\)|<p>\1</p>\n          <ul>|' | sed 's|\* \(.*\)|            <li>\1</li>|' | (tee -a && echo '          </ul>'))"
PROJECT_DESCRIPTION="$(sed -n '/<!-- project_description_start -->/,/<!-- project_description_end -->/p' ../README.md | sed -e 's/<\/\?\(b\|a[^>]*\|br\)>//g' | sed '1d;$d')"
# The jinja2 module renders it rather than the jinja2 command of jinja2-cli, which
# the devcontainer image lacks while its Python carries the module. The
# environment is the one the command sets up.
PROJECT_DESCRIPTION="$PROJECT_DESCRIPTION" VERSION="$(tr -d '\n' < ../VERSION.txt)" DATE="$(date --iso)" \
CHANGELOG_CONTENT="$CHANGELOG_CONTENT" python3 -c '
import os, sys, jinja2
environment = jinja2.Environment(loader=jinja2.FileSystemLoader("files"), keep_trailing_newline=True)
sys.stdout.write(environment.get_template("org.squey.Squey.metainfo.xml.j2").render(
    project_description=os.environ["PROJECT_DESCRIPTION"], version=os.environ["VERSION"],
    date=os.environ["DATE"], changelog=os.environ["CHANGELOG_CONTENT"]))
' > files/org.squey.Squey.metainfo.xml

if [ "$APPSTREAM_LINTER" = true ]; then
  flatpak-builder-lint appstream files/org.squey.Squey.metainfo.xml
fi