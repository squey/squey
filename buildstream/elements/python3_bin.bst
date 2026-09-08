kind: manual

(?):
- (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
    sources:
    - kind: remote
      url: python:ftp/python/3.14.7/python-3.14.7-macos11.pkg
      ref: 70c5239ad2d62925d2947e46921d0ddd3d35be3d2f0a2d50db33da507dbcb419
- target_triple == "x86_64-w64-mingw32":
    sources:
    - kind: remote
      url: msys2:mingw/mingw64/mingw-w64-x86_64-python-3.14.7-1-any.pkg.tar.zst
      ref: 39aa1fe66f065490843c633d0819e4548718ed99b7bdf1b02512a59f31afa474

(@): buildstream/elements/base/target.yml

variables:
  python-framework-payload-path: "Python_Framework.pkg/Payload"
  python-framework-install-dir: "%{install-root}/%{libdir}/Python.framework"

config:
    (?):
    - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
        install-commands:
        - |
          xar -x -f *.pkg "%{python-framework-payload-path}"
          gunzip < "%{python-framework-payload-path}" | cpio -idm
          mkdir -p "%{python-framework-install-dir}"
          mv Versions Resources Python "%{python-framework-install-dir}"
          cd "%{install-root}/%{libdir}" && ln -s Python.framework/Versions/Current/lib/libpython3.*.dylib && cd -
          cd "%{install-root}/%{libdir}"
          rm -rf Python.framework/Versions/Current/lib/libtk* # Useless and using non-public or deprecated macOS API that prevents publishing on the App Store
          rm -rf Python.framework/Versions/Current/lib/python*/lib-dynload/_tkinter*
          rm -rf Python.framework/Versions/Current/Frameworks/Tcl.framework Python.framework/Versions/Current/Frameworks/Tk.framework
          sed "s/, 'itms-services'//" -i Python.framework/Versions/Current/lib/python*/urllib/parse.py && rm -rf Python.framework/Versions/Current/lib/python*/test # Remove 'itms-services' occurences
          find "Python.framework" -type f \( -name "*.dylib" -o -name "Python" \) \
            -exec echo "{}" \; \
            -exec %{toolchain-dir}/%{host}-lipo -extract "%{macho-arch}" "{}" -output "{}" \; \
            -exec %{toolchain-dir}/%{host}-codesign_allocate -r -i "{}" -o "{}" \; \
            -exec %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/{}" "{}" \;
          cp -r Python.framework/Versions/Current/include/* %{install-root}/%{includedir}
    - target_triple == "x86_64-w64-mingw32":
        install-commands:
        - |
          tar --no-same-owner --zstd -xvf *.zst
          mkdir -p "%{install-root}%{bindir}"
          mkdir -p "%{install-root}%{libdir}"
          mkdir -p "%{install-root}%{includedir}"
          mv mingw64/include/python%{python-version}/* "%{install-root}%{includedir}"
          mv mingw64/bin/* "%{install-root}%{bindir}"
          rm -rf mingw64/lib/python%{python-version}/test
          find mingw64/lib/python%{python-version}/ -regex '^.*\(__pycache__\|\.py[co]\)$' -delete
          mv mingw64/lib/* "%{install-root}%{libdir}"

          # mkdir -p "%{install-root}%{bindir}/python"
          # mkdir -p "%{install-root}%{libdir}"
          # unzip *.nupkg "tools/*"
          # mv tools/* "%{install-root}%{bindir}/python"
          # mv %{install-root}%{bindir}/python/include* "%{install-root}%{includedir}"
          # mv %{install-root}%{bindir}/python/libs/* "%{install-root}%{libdir}"

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/libarchive.bst
- zstd.bst
