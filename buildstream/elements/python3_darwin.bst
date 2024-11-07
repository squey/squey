kind: manual

sources:
- kind: remote
  url: python:ftp/python/3.12.8/python-3.12.8-macos11.pkg
  ref: c411b5372d563532f5e6b589af7eb16e95613d61bd5af7bfe78563467130bbff

(@): buildstream/elements/base/target.bst

variables:
  python-framework-payload-path: "Python_Framework.pkg/Payload"
  python-framework-install-dir: "%{install-root}/%{libdir}/Python.framework"

config:
    install-commands:
    - |
      xar -x -f *.pkg "%{python-framework-payload-path}"
      gunzip < "%{python-framework-payload-path}" | cpio -idm
      mkdir -p "%{python-framework-install-dir}"
      mv "Versions" "%{python-framework-install-dir}"
      cd "%{install-root}/%{libdir}" && ln -s "Python.framework/Versions/Current/lib/libpython3.12.dylib" && cd -

      cd "%{install-root}/%{libdir}"
      find "Python.framework" \( -name "*.dylib" -o -name "Python" \) \
        -exec echo "{}" \; \
        -exec %{toolchain-dir}/%{host}-lipo -extract "%{macho-arch}" "{}" -output "{}" \; \
        -exec %{toolchain-dir}/%{host}-codesign_allocate -r -i "{}" -o "{}" \; \
        -exec %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/{}" "{}" \;

      mv Python.framework/Versions/Current/include/* %{install-root}/%{includedir}

build-depends:
- sdk.bst