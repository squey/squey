kind: manual

sources:
- kind: local
  path: buildstream/files/macos_bundle
  directory: buildstream/files/macos_bundle
- kind: local
  path: VERSION.txt

(@):
    - buildstream/elements/base/target.bst
    - buildstream/elements/macos_bundle/config.yml

variables:
    bundle-name: "Squey.app"

config:
  install-commands:
  - |
    mkdir -p "%{install-root}/bundle_root/%{bundle-name}/Contents" && cd "$_"
    mkdir MacOS Frameworks PlugIns Resources
    cp -r %{bindir}/* MacOS
    cp -r %{libdir}/* Frameworks
    cp -r %{libdir}/squey/plugins/* PlugIns
    cp -r %{libdir}/qt6/plugins/* PlugIns
    mkdir share
    cp -r %{prefix}/share/pocl share
    cp -r %{prefix}/share/squey share
    cp %{build-root}/buildstream/files/macos_bundle/squey.icns Resources
    VERSION=$(cat %{build-root}/VERSION.txt)
    CURRENT_YEAR=$(date +%Y)
    sed -e "s/\$VERSION/$VERSION/g" -e "s/\$CURRENT_YEAR/$CURRENT_YEAR/g" %{build-root}/buildstream/files/macos_bundle/Info.plist > Info.plist
    ln -s /Applications "%{install-root}/bundle_root/Applications"

build-depends:
- squey-cleanup.bst
- sdk.bst