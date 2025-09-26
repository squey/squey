#!/bin/bash

## Renewing an Apple certificate:
# openssl genrsa -out cert.key 2048
# openssl req -new -key cert.key -out cert.csr -subj "/C=FR/O=XXXXXXXX/CN=XXX XXX: COMPANY (TEAM_ID)"
# openssl pkcs12 -legacy -export -out cert.p12 -inkey cert.key -in cert.cer

set -e

PACKAGEDIR_X86="export/x86_64-apple-darwin"
PACKAGEDIR_ARM="export/aarch64-apple-darwin"
APPNAME="Squey"
BUNDLE_ID="org.squey.Squey"
BUNDLENAME_ARM="${APPNAME}-arm64.app"
BUNDLENAME_X86="${APPNAME}-x86_64.app"
DMGNAME_ARM="${BUNDLENAME_ARM%.app}.dmg"
DMGNAME_X86="${BUNDLENAME_X86%.app}.dmg"
BUNDLENAME="${APPNAME}.app"
PKGNAME="${APPNAME}.pkg"
MOUNTED_VOLUME="/Volumes/Squey"
MOUNTED_BUNDLEDIR="$MOUNTED_VOLUME/$BUNDLENAME"
KEYCHAINNAME="tmp.keychain"
KEYCHAINPATH=$HOME/Library/Keychains/${KEYCHAINNAME}-db
APPLE_AUTH_KEY_PATH="apple_auth_key.p8"
echo -ne "${APPLE_AUTH_KEY//\\n/$'\n'}" > "$APPLE_AUTH_KEY_PATH"
sha256sum "$APPLE_AUTH_KEY_PATH"

# Convert packages to universal binary
hdiutil attach -nobrowse $PACKAGEDIR_ARM/*.dmg
ditto "$MOUNTED_BUNDLEDIR" "$BUNDLENAME_ARM"
hdiutil detach "$MOUNTED_VOLUME"
hdiutil attach -nobrowse $PACKAGEDIR_X86/*.dmg
ditto "$MOUNTED_BUNDLEDIR" "$BUNDLENAME_X86"
hdiutil detach "$MOUNTED_VOLUME"
hdiutil attach -nobrowse $PACKAGEDIR_ARM/*.dmg
ditto "$MOUNTED_BUNDLEDIR" "$BUNDLENAME"
hdiutil detach "$MOUNTED_VOLUME"
hdiutil attach -nobrowse $PACKAGEDIR_X86/*.dmg
ditto "$MOUNTED_BUNDLEDIR" "$BUNDLENAME_X86"
hdiutil detach "$MOUNTED_VOLUME"
find "$BUNDLENAME" -type f | while read arm_file; do
    if file "$arm_file" | grep -q "Mach-O"; then
        rel_path="${arm_file#$BUNDLENAME/}"
        x86_file="$BUNDLENAME_X86/$rel_path"
        out_file="$BUNDLENAME/$rel_path"
        if [ -f "$x86_file" ]; then
            mkdir -p "$(dirname "$out_file")"
            lipo -create "$x86_file" "$arm_file" -output "$out_file" || true
        else
            echo "Missing x86_64 file: $x86_file" >&2
        fi
    fi
done

# Create unique internal version number
APP_VERSION=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" $BUNDLENAME/Contents/Info.plist)
APP_VERSION_MINOR="${APP_VERSION%.*}"
BUNDLE_VERSION="$APP_VERSION_MINOR.$(date +%s)"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $BUNDLE_VERSION" $BUNDLENAME/Contents/Info.plist

# Create temporary keychain
for cert in APPLE_DISTRIBUTION_CERT_B64 APPLE_INSTALLER_CERT_B64 APPLE_DEVELOPER_CERT_B64; do
    CERT_NAME="${cert}.p12"
    base64 -d -i "${!cert}" -o "$CERT_NAME"
done
security delete-keychain "$KEYCHAINNAME" &> /dev/null || true
security create-keychain -p "" "$KEYCHAINNAME"
security unlock-keychain -p "" "$KEYCHAINNAME"
security list-keychains -d user -s "$KEYCHAINPATH" "$HOME/Library/Keychains/login.keychain-db"
security default-keychain -d user -s "$KEYCHAINPATH"

# Download and add Apple certificates
ALLOWED_TOOLS=" \
    -T /usr/bin/codesign \
    -T /usr/bin/productbuild"
curl -sLO https://www.apple.com/certificateauthority/AppleRootCA-G3.cer
curl -sLO https://www.apple.com/certificateauthority/AppleWWDRCAG3.cer
curl -sLO https://www.apple.com/certificateauthority/AppleWWDRCAG4.cer
curl -sLO https://www.apple.com/certificateauthority/DeveloperIDG2CA.cer
for cert in AppleRootCA-G3.cer AppleWWDRCAG3.cer AppleWWDRCAG4.cer DeveloperIDG2CA.cer; do
    security import "$cert" -k "$KEYCHAINNAME" $ALLOWED_TOOLS
    sleep 1 # this is ridiculously important
done
for cert in *.p12; do
    security import "$cert" -k "$KEYCHAINNAME" -P "$APPSTORE_CERT_PASSPHRASE" $ALLOWED_TOOLS
done
security set-key-partition-list -S apple-tool:,apple:,codesign:,system: -s -k "" "$KEYCHAINPATH"
sha256sum *.cer
security find-identity -v
security find-certificate -a

codesign_retry() {
  cmd=( "$@" )
  max=5
  delay=60
  CODESIGN_LOG_FILE="/tmp/codesign_last_err.log"
  for i in $(seq 1 $max); do
    echo "Attempt $i: ${cmd[*]}"
    if "${cmd[@]}" 2> "$CODESIGN_LOG_FILE"; then
      return 0
    fi
    if grep -q "timestamp service is not available" "$CODESIGN_LOG_FILE"; then
      echo "Timestamp service error — retrying after $delay s"
      sleep $delay
      delay=$((delay * 2))
      continue
    else
      cat "$CODESIGN_LOG_FILE" >&2
      return 1
    fi
  done
  return 1
}

sign()
{
    bundlename="$1"
    CERT_IDENTITY="$2"

    # Sign binaries
    find "$bundlename/Contents" -type f | while read bin; do
        TYPE=$(file "$bin")
        if echo "$TYPE" | grep -q "Mach-O"; then
            codesign_retry codesign --verbose=4 --display --keychain "$KEYCHAINPATH" --deep --force --options runtime --sign "$CERT_IDENTITY" "$bin"
        fi
    done

    # Sign executables with sandbox entitlement
    EXECUTABLES=$(find "$bundlename" -type f -perm +111 | while read -r f; do
        TYPE=$(file "$f")
        if echo "$TYPE" | grep -q "Mach-O .*executable"; then
            echo "$f"
        fi
    done)
    for exe in $EXECUTABLES; do
        codesign_retry codesign --verbose=4 --display --keychain "$KEYCHAINPATH" --deep --force --entitlements buildstream/files/macos_bundle/sandbox_entitlement.plist --options runtime --sign "$CERT_IDENTITY" "$exe"
    done

    # Sign frameworks
    find "$bundlename/Contents/Frameworks" -type d -name "*.framework" | while read fw; do
        codesign_retry codesign --verbose=4 --display --keychain "$KEYCHAINPATH" --deep --force --options runtime --sign "$CERT_IDENTITY" "$fw"
    done
}

# Ping "timestamp.apple.com" in the background (see https://stackoverflow.com/a/71013831/340754)
ping timestamp.apple.com &>/dev/null &

# Sign packages for external distribution
sign_and_notarize()
{
    NOARCH_BUNDLENAME="$BUNDLENAME"
    bundlename="$1"
    CERT_IDENTITY="$2"
    DMGNAME="${bundlename%.app}.dmg"
    ROOT_DIR="${bundlename}_root"
    sign "$bundlename" "$CERT_IDENTITY"
    mkdir -p "$ROOT_DIR" "$CI_PROJECT_DIR/public"
    mv $bundlename $ROOT_DIR/$NOARCH_BUNDLENAME
    hdiutil create -volname "$APPNAME" \
        -srcfolder "$ROOT_DIR" \
        -format ULMO \
        -ov \
        "$CI_PROJECT_DIR/public/$DMGNAME"
    codesign --verbose=4 --display --keychain "$KEYCHAINPATH" --deep --force --options runtime --sign "$CERT_IDENTITY" "$CI_PROJECT_DIR/public/$DMGNAME"
    xcrun notarytool submit "$CI_PROJECT_DIR/public/$DMGNAME" --wait --timeout 10m \
        --key "$APPLE_AUTH_KEY_PATH" \
        --key-id "$APPLE_KEY_ID" \
        --issuer "$APPLE_ISSUER_ID"
    xcrun stapler staple "$CI_PROJECT_DIR/public/$DMGNAME"
    rm -rf "$ROOT_DIR"
}
sign_and_notarize "$BUNDLENAME_ARM" "$APPLE_DEVELOPER_CERT_IDENTITY"
sign_and_notarize "$BUNDLENAME_X86" "$APPLE_DEVELOPER_CERT_IDENTITY"

# Sign and deliver the app to the App Store
SUBMISSION_INFO='{"export_compliance_uses_encryption": true,
                  "export_compliance_compliance_required": true,
                  "export_compliance_encryption_updated": false,
                  "export_compliance_contains_third_party_cryptography": true,
                  "export_compliance_contains_proprietary_cryptography": false,
                  "export_compliance_available_on_french_store": true,
                  "export_compliance_is_exempt": false,
                  "add_id_info_uses_idfa": false }'
sign "$BUNDLENAME" "$APPLE_DISTRIBUTION_CERT_IDENTITY"
codesign --verbose=4 --display --keychain "$KEYCHAINPATH" --deep --force --entitlements buildstream/files/macos_bundle/sandbox_entitlement.plist --options runtime --sign "$APPLE_DISTRIBUTION_CERT_IDENTITY" "$BUNDLENAME"
productbuild --component "$BUNDLENAME" /Applications --sign "$APPLE_INSTALLER_CERT_IDENTITY" "$PKGNAME"
fastlane deliver --force --pkg "$PKGNAME" --app_identifier "$BUNDLE_ID" --api_key_path "$APPLE_API_KEY_JSON" --skip_screenshots --skip_metadata --run_precheck_before_submit false --submission_information "$SUBMISSION_INFO" --submit_for_review || true
