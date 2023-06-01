kind: manual

sources:
  - kind: local
    path: buildstream/files

build-depends:
- base.bst
# - squey-cleanup.bst

config:
  install-commands:
  - install -D -m 0644 "org.squey.Squey.png" -t "%{install-root}%{prefix}/share/icons/hicolor/128x128/apps/"
  - install -D -m 0644 "org.squey.Squey.png" -t "%{install-root}%{prefix}/share/app-info/icons/flatpak/128x128/"
  - |
    cat <<EOF >org.squey.Squey.desktop
    [Desktop Entry]
    Type=Application
    Name=Squey
    Exec=squey_launcher.sh
    Icon=org.squey.Squey
    Categories=Science,Graphics;Network;
    EOF
    install -D -m 0644 "org.squey.Squey.desktop" -t "%{install-root}%{prefix}/share/applications"
  - install -D -m 0644 "org.squey.Squey.metainfo.xml" -t "%{install-root}%{prefix}/share/metainfo"
