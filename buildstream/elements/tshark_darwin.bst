kind: manual
description: |

    TShark

(?):
  - target_triple == "x86_64-apple-darwin":
      sources:
        - kind: remote
          url: wireshark:osx/Wireshark%204.4.2%20Intel%2064.dmg
          ref: 5f379065fa16424c68362346177b146b6050fcb77f2468b85cf197f09e399113
          filename: wireshark.dmg
  - target_triple == "aarch64-apple-darwin":
      sources:
        - kind: remote
          url: wireshark:osx/Wireshark%204.4.2%20Arm%2064.dmg
          ref: e60c577e9e2ffff7b2fc10d50c27f41a061d3140ae3bc2a223dba882da00c428
          filename: wireshark.dmg
  
(@): buildstream/elements/base/target.bst

config:
  configure-commands:
  - |
    dmg2img wireshark.dmg
    7zz x wireshark.img Wireshark*/Wireshark.app/*
    cd Wireshark*/Wireshark.app/Contents
    install -D Frameworks/*.dylib -t "%{install-root}%{libdir}"
    install -m 755 -D MacOS/tshark -t "%{install-root}%{bindir}"

build-depends:
- sdk.bst
- macos_bundle/dmg2img.bst
- macos_bundle/7zip.bst