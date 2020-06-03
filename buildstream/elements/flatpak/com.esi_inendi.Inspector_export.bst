kind: import

sources:
  - kind: local
    path: buildstream/files

kind: manual

build-depends:
- base.bst
#- inendi-inspector-cleanup.bst

config:
  install-commands:
  - install -D -m 0644 "pcap-inspector_launcher.sh" -t "%{install-root}%{prefix}/bin/"
  - install -D -m 0644 "com.pcap_inspector.Inspector.png" -t "%{install-root}/export/share/icons/hicolor/64x64/apps/"
  - |
    cat <<EOF >com.pcap_inspector.Inspector.desktop
    [Desktop Entry]
    Type=Application
    Name=PCAP Inspector
    Exec=inspector_launcher.sh
    Icon=com.pcap_inspector.Inspector
    Categories=Graphics;Network;
    EOF
    install -D -m 0644 -t "%{install-root}/export/share/applications" com.pcap_inspector.Inspector.desktop
