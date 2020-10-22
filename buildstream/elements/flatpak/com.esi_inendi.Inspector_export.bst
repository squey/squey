kind: import

sources:
  - kind: local
    path: buildstream/files

kind: manual

build-depends:
- base.bst
# - inendi-inspector-cleanup.bst

config:
  install-commands:
  - install -D -m 0644 "com.esi_inendi.Inspector.png" -t "%{install-root}/export/share/icons/hicolor/64x64/apps/"
  - |
    cat <<EOF >com.esi_inendi.Inspector.desktop
    [Desktop Entry]
    Type=Application
    Name=INENDI Inspector
    Exec=inspector_launcher.sh
    Icon=com.esi_inendi.Inspector
    Categories=Graphics;Network;
    EOF
    install -D -m 0644 -t "%{install-root}/export/share/applications" com.esi_inendi.Inspector.desktop
