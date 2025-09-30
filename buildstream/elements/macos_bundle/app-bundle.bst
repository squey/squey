kind: manual

sources:
- kind: local
  path: buildstream/files/macos_bundle
  directory: buildstream/files/macos_bundle
- kind: local
  path: VERSION.txt

(@):
    - buildstream/elements/base/target.yml
    - buildstream/elements/macos_bundle/config.yml

variables:
    bundle-name: "Squey.app"

config:
  install-commands:
  - |
    mkdir -p "%{install-root}/bundle_root/%{bundle-name}/Contents" && cd "$_"
    mkdir MacOS Frameworks PlugIns Resources
    cp -ar %{bindir}/* MacOS
    cp -ar %{libdir}/* Frameworks
    cp -ar %{libdir}/squey/plugins/* PlugIns
    cp -ar %{libdir}/qt6/plugins/* PlugIns
    mv Frameworks/python* Resources/python
    rm -rf Frameworks/{objects-Release,sigc++-3.0,icu,postgresql/pgxs}
    find . -name "libclang_rt.*" -exec rm -f {} \;
    sed 's/org.python.python/org.squey.python/' -i Frameworks/Python.framework/Versions/Current/Resources/Python.app/Contents/Info.plist
    mkdir share
    cp -ar %{prefix}/share/pocl share
    cp -ar %{prefix}/share/squey share
    cp %{build-root}/buildstream/files/macos_bundle/squey.icns Resources
    VERSION=$(cat %{build-root}/VERSION.txt)
    CURRENT_YEAR=$(date +%Y)
    sed -e "s/\$VERSION/$VERSION/g" -e "s/\$CURRENT_YEAR/$CURRENT_YEAR/g" %{build-root}/buildstream/files/macos_bundle/Info.plist > Info.plist
    ln -s /Applications "%{install-root}/bundle_root/Applications"

build-depends:
- squey-cleanup.bst
- sdk.bst
