kind: manual

(?):
- target_triple == "x86_64-w64-mingw32":
    sources:
    - kind: remote
      url: msys2:mingw/mingw64/mingw-w64-x86_64-python-pip-25.0-1-any.pkg.tar.zst
      ref: a9bea0aa0e020a9543d42e8798835c9a9292f7f44b30b7588523d4d074562fac

(@): buildstream/elements/base/target.bst

variables:
    python-site-package: "python3.12/site-packages"

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
          mv mingw64/lib/%{python-site-package}/* "%{install-root}%{libdir}/%{python-site-package}"

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/unzip.bst
- zstd.bst

depends:
- python3_bin.bst