kind: manual
description: |

    TShark

(?):
  - target_triple == "x86_64-apple-darwin":
      sources:
        - kind: remote
          url: wireshark:download/osx/all-versions/Wireshark%204.4.9%20Intel%2064.dmg
          ref: f7e742a6cd42f13c81ad63a99764262c7e15bd66c8b48eb9bf879803150d5b7d
          filename: wireshark.dmg
  - target_triple == "aarch64-apple-darwin":
      sources:
        - kind: remote
          url: wireshark:download/osx/all-versions/Wireshark%204.4.9%20Arm%2064.dmg
          ref: 6623f789646c6a64735d2179fd9333a05606f83f7946d897448eea2074e8654b
          filename: wireshark.dmg
  - target_triple == "x86_64-w64-mingw32":
      sources:
        - kind: remote
          url: wireshark:download/win64/all-versions/WiresharkPortable64_4.4.9.paf.exe
          ref: 8825a080a5337cecebba2127dc894662bd265bc47ba1c4841b0253126707abf1
          filename: wireshark.exe

(@): buildstream/elements/base/target.yml

config:
  (?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      configure-commands:
        - |
          dmg2img wireshark.dmg
          7zzs x wireshark.img Wireshark*/Wireshark.app/*
          cd Wireshark*/Wireshark.app/Contents
          install -D Frameworks/*.dylib -t "%{install-root}%{libdir}"
          install -m 755 -D MacOS/tshark -t "%{install-root}%{bindir}"
  - target_triple == "x86_64-w64-mingw32":
      configure-commands:
        - |
          7zzs x -y wireshark.exe App/Wireshark/*
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
