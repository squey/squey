kind: flatpak_image

depends:
- filename: flatpak/flatpak-image.bst
  type: build
- filename: flatpak/org.squey.Squey_export.bst
  type: build

config:
  directory: '%{prefix}'

  # buildstream/scripts/export_flatpak.sh adds the command and the permissions, from
  # buildstream/files/flatpak/finish_args.sh.
  metadata:
    Application:
      name: org.squey.Squey
      runtime: org.freedesktop.Platform/%{gcc_arch}/%{runtime-branch}
      sdk: org.freedesktop.Sdk/%{gcc_arch}/%{runtime-branch}
      branch: "%{branch}"

    'Extension org.squey.Squey.Debug':
      directory: lib/debug
