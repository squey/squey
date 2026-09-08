kind: manual

(?):
- target_triple == "x86_64-w64-mingw32":
    sources:
    - kind: remote
      url: msys2:mingw/mingw64/mingw-w64-x86_64-python-pip-26.2.1-1-any.pkg.tar.zst
      ref: 04ec74a0ffa10b97fcceea48969609e5c72a1fa34e992fadc168c654519be718

(@): buildstream/elements/base/target.yml

variables:
    python-site-package: "python%{python-version}/site-packages"

config:
    (?):
    - target_triple == "x86_64-w64-mingw32":
        install-commands:
        - |
          tar --no-same-owner --zstd -xvf *.zst
          mkdir -p "%{install-root}%{bindir}"
          mkdir -p "%{install-root}%{libdir}/%{python-site-package}"
          mv mingw64/bin/* "%{install-root}%{bindir}"
          find mingw64 | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
          mv mingw64/lib/python*/site-packages/* "%{install-root}%{libdir}/%{python-site-package}"

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/libarchive.bst
- zstd.bst

depends:
- python3_bin.bst
