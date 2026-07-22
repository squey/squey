kind: manual
description: |

    TShark

(?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      sources:
        - kind: remote
          url: wireshark:download/osx/all-versions/Wireshark%204.6.4.dmg
          ref: 08150f79cfc5828820f991b6d944c68536db9595b1c3052982bbde79fb2053df
          filename: wireshark.dmg
  - target_triple == "x86_64-w64-mingw32":
      sources:
        - kind: remote
          url: wireshark:download/win64/all-versions/WiresharkPortable64_4.6.4.paf.exe
          ref: 2f144183e24d14c3573053b85e304819a5409399383a3f3f6b2feac5e36d1dbb
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
