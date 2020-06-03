kind: flatpak_image

depends:
- filename: flatpak/flatpak-image.bst
  type: build
- filename: flatpak/com.pcap_inspector.Inspector_export.bst
  type: build

config:
  directory: '%{prefix}'
  export: 'export'

  metadata:
    Application:
      name: com.pcap_inspector.Inspector
      base: com.esi_inendi.Inspector
      runtime: org.freedesktop.Platform/%{gcc_arch}/%{runtime-branch}
      sdk: org.freedesktop.Sdk/%{gcc_arch}/%{runtime-branch}
      command: pcap-inspector_launcher.sh
      branch: "%{branch}"

    Context:
      shared: "network;ipc;"
      sockets: "x11;wayland;pulseaudio;"
      devices: "all;"
      filesystems: "host;xdg-download;/opt/inendi;~/.inendi;/tmp;"

    'Extension com.esi_inendi.Inspector.Debug':
      directory: lib/debug
