kind: flatpak_image

depends:
- filename: flatpak/flatpak-image.bst
  type: build
- filename: flatpak/org.squey.Squey_export.bst
  type: build

config:
  directory: '%{prefix}'

  metadata:
    Application:
      name: org.squey.Squey
      runtime: org.freedesktop.Platform/%{gcc_arch}/%{runtime-branch}
      sdk: org.freedesktop.Sdk/%{gcc_arch}/%{runtime-branch}
      command: squey_launcher.sh
      branch: "%{branch}"

    Context:
      shared: "network;ipc;"
      sockets: "x11;wayland;"
      devices: "dri;"
      filesystems: "host;"
      features: "devel"

    'Extension org.squey.Squey.Debug':
      directory: lib/debug
