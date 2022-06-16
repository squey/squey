kind: manual

sources:
  - kind: local
    path: buildstream/files

build-depends:
- base.bst
# - inendi-inspector-cleanup.bst

config:
  install-commands:
  - install -D -m 0644 "com.gitlab.inendi.Inspector.png" -t "%{install-root}/export/share/icons/hicolor/128x128/apps/"
  - |
    cat <<EOF >com.gitlab.inendi.Inspector.desktop
    [Desktop Entry]
    Type=Application
    Name=INENDI Inspector
    Exec=inspector_launcher.sh
    Icon=com.gitlab.inendi.Inspector
    Categories=Graphics;Network;
    EOF
    install -D -m 0644 "com.gitlab.inendi.Inspector.desktop" -t "%{install-root}/export/share/applications"
  - install -D -m 0644 "com.gitlab.inendi.Inspector.metainfo.xml" -t "%{install-root}/app/share/metainfo"
