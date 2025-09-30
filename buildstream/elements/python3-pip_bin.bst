kind: manual

(?):
- target_triple == "x86_64-w64-mingw32":
    sources:
    - kind: remote
      url: msys2:mingw/mingw64/mingw-w64-x86_64-python-pip-25.2-1-any.pkg.tar.zst
      ref: 4d678225ad8c42765e1fbca05ea43f5c8fb56d77b1379948af2837c4cae37d6c

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
- freedesktop-sdk.bst:components/unzip.bst
- zstd.bst

depends:
- python3_bin.bst
