kind: manual
description: |

    TShark

(?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      sources:
        - kind: remote
          url: wireshark:download/osx/all-versions/Wireshark%204.6.8.dmg
          ref: 7de945ed1ba324259ba7e3b2ca2fe11a854cf48a33dc6d4423dd531e466a1f3a
          filename: wireshark.dmg
  - target_triple == "x86_64-w64-mingw32":
      sources:
        - kind: remote
          url: wireshark:download/win64/all-versions/WiresharkPortable64_4.6.8.paf.exe
          ref: d6cf5056586a0156c8634a9d9b3097e60a8dda408b889c9afc4445ec2b18d581
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
          # Upstream ships a universal binary: keep only the slice we are building for
          find "%{install-root}%{libdir}" "%{install-root}%{bindir}" -type f \( -name "*.dylib" -o -name "tshark" \) \
            -exec %{toolchain-dir}/%{host}-lipo -extract "%{macho-arch}" "{}" -output "{}" \; \
            -exec %{toolchain-dir}/%{host}-codesign_allocate -r -i "{}" -o "{}" \;
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
