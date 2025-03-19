VERSION=$(cat VERSION.txt)
PACKAGE_NAME="squey_${VERSION}.dmg"

genisoimage -V "Squey" -D -R -apple -o "$PACKAGE_NAME.uncompressed" "/bundle_root"
dmg "$PACKAGE_NAME.uncompressed" "/output/$PACKAGE_NAME"

cd /mac/tests && zip -r /output/testsuite.zip .