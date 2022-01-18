kind: manual

sources:
  - kind: local
    path: buildstream/files

build-depends:
- base.bst
# - inendi-inspector-cleanup.bst

config:
  install-commands:
  - install -D -m 0644 "org.inendi.Inspector.png" -t "%{install-root}/export/share/icons/hicolor/64x64/apps/"
  - |
    cat <<EOF >org.inendi.Inspector.desktop
    [Desktop Entry]
    Type=Application
    Name=INENDI Inspector
    Exec=inspector_launcher.sh
    Icon=org.inendi.Inspector
    Categories=Graphics;Network;
    EOF
    install -D -m 0644 -t "%{install-root}/export/share/applications" org.inendi.Inspector.desktop
