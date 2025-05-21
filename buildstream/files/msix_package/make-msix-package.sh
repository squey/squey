#!/bin/bash

set -e

CONFIG_FILE="buildstream/files/msix_package/config.yml"
PACKAGE_ROOT=$(yq '.variables."package-root"' $CONFIG_FILE)
MSIX_PROJECT="AppxManifest.xml"
VERSION=$(cat VERSION.txt)
PACKAGE_NAME="squey_${VERSION}.msix"

cp -ar "/$PACKAGE_ROOT" .
sed -e "s|{{version}}|${VERSION}|" "buildstream/files/msix_package/$MSIX_PROJECT.j2" > "$PACKAGE_ROOT/$MSIX_PROJECT"
makemsix pack -d "$PACKAGE_ROOT" -p "/output/${PACKAGE_NAME}"

cd /win/tests && 7zz a /output/testsuite.zip .