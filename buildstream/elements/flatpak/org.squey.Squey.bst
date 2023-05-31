kind: flatpak_image

depends:
- filename: flatpak/flatpak-image.bst
  type: build
- filename: flatpak/org.squey.Squey_export.bst
  type: build

config:
  directory: '%{prefix}'
  export: 'export'

  metadata:
    Application:
      name: org.squey.Squey
      runtime: org.freedesktop.Platform/%{gcc_arch}/%{runtime-branch}
      sdk: org.freedesktop.Sdk/%{gcc_arch}/%{runtime-branch}
      command: squey_launcher.sh
      branch: "%{branch}"

    Context:
      shared: "network;ipc;"
      sockets: "x11;wayland;pulseaudio;"
      devices: "all;"
      filesystems: "host;xdg-download;/opt/squey;~/.squey;/tmp;"
      features: "devel"

    'Extension org.squey.Squey.Debug':
      directory: lib/debug
