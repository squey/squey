kind: manual
description: |

    TShark

(?):
  - target_triple == "x86_64-apple-darwin":
      sources:
        - kind: remote
          url: wireshark:download/osx/all-versions/Wireshark%204.4.2%20Intel%2064.dmg
          ref: 5f379065fa16424c68362346177b146b6050fcb77f2468b85cf197f09e399113
          filename: wireshark.dmg
  - target_triple == "aarch64-apple-darwin":
      sources:
        - kind: remote
          url: wireshark:download/osx/all-versions/Wireshark%204.4.2%20Arm%2064.dmg
          ref: e60c577e9e2ffff7b2fc10d50c27f41a061d3140ae3bc2a223dba882da00c428
          filename: wireshark.dmg
  - target_triple == "x86_64-w64-mingw32":
      sources:
        - kind: remote
          url: wireshark:download/win64/all-versions/WiresharkPortable64_4.4.2.paf.exe
          ref: 937de59eb2ca8d9ec706e3b218f780ca6f3e9a209cb32a34e4557e381a17c895
          filename: wireshark.exe

(@): buildstream/elements/base/target.bst

config:
  (?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      configure-commands:
        - |
          dmg2img wireshark.dmg
          7zz x wireshark.img Wireshark*/Wireshark.app/*
          cd Wireshark*/Wireshark.app/Contents
          install -D Frameworks/*.dylib -t "%{install-root}%{libdir}"
          install -m 755 -D MacOS/tshark -t "%{install-root}%{bindir}"
  - target_triple == "x86_64-w64-mingw32":
      configure-commands:
        - |
          7zz x -y wireshark.exe App/Wireshark/*
          cd App/Wireshark
          DEST_DIR="%{install-root}%{bindir}/tshark"
          mkdir -p "${DEST_DIR}/plugins"
          install -m 755 -D tshark.exe -t "${DEST_DIR}"
          cp -r plugins/*.* -t "${DEST_DIR}/plugins"

          # Dynamically copy local dependencies
          extract_local_deps() {
            local file="$1"
            local deps=$(%{host}-objdump -p "$file" 2>/dev/null | grep 'DLL Name' | awk '{print $3}')
            for dep in $deps; do
                if [[ -f "./$dep" && ! " ${seen[*]} " =~ " $dep " ]]; then
                    seen+=("$dep")
                    deps_list+=" $dep"
                    extract_local_deps "./$dep"
                fi
            done
          }
          deps_list=""
          declare -a seen
          extract_local_deps "tshark.exe"
          cp $deps_list "$DEST_DIR"

build-depends:
- sdk.bst
- 7zip_host.bst
- macos_bundle/dmg2img.bst
