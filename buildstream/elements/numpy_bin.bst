kind: manual
description: |

    NumPy is the fundamental package needed for scientific computing with Python.

(?):
  - target_triple == "x86_64-apple-darwin":
      sources:
        - kind: remote
          url: pythonhosted:packages/ac/f8/c3b222bf075b50afd8e949a07a15c4b312a4a84bd8102a332bcd953cbbb4/numpy-2.5.2-cp314-cp314-macosx_10_15_x86_64.whl
          ref: d787cf769c3baeb5f6235e778edb52c08dfa923789b5958f28e6450f96107cb1
  - target_triple == "aarch64-apple-darwin":
      sources:
        - kind: remote
          url: pythonhosted:packages/17/e1/2c1d4b1987795a92b5bbf7c24fe249ab96aa2573ab0d7604802c189d7b86/numpy-2.5.2-cp314-cp314-macosx_11_0_arm64.whl
          ref: 24b9dc2e3d84aa58523798805194e23e736f3f6ce2d1a5b92583ae734e6dbda8
  - target_triple == "x86_64-w64-mingw32":
      sources:
        - kind: remote
          url: msys2:mingw/mingw64/mingw-w64-x86_64-python-numpy-2.5.2-1-any.pkg.tar.zst
          ref: c5d5d0d9172ca2cfdbcfeef0301153576590dfdb11b2e0fb0a276e3ef2c240b4
        - kind: remote
          url: msys2:mingw/mingw64/mingw-w64-x86_64-openblas-0.3.34-1-any.pkg.tar.zst
          ref: cac7623e6b0f07a7ad72fa2e6d4e346e9025fc14d0ccb68d0b79a50765f05e88
        - kind: remote
          url: msys2:mingw/mingw64/mingw-w64-x86_64-gcc-libgfortran-16.2.0-3-any.pkg.tar.zst
          ref: bc9095f979fb555e116bd7cab7a514e65988bb9d7e4ddb3f333acd1a5295936d

(@): buildstream/elements/base/target.yml

variables:
    python-site-packages-dir: "%{install-root}%{libdir}/python%{python-version}/site-packages"

config:
  install-commands:
    - |
      mkdir -p "%{python-site-packages-dir}"
      if [ "%{target-platform}" == "darwin" ]; then
        unzip *.whl 'numpy/*' -d "%{python-site-packages-dir}"
      elif [ "%{target-platform}" == "windows" ]; then
        mkdir -p "%{install-root}%{bindir}"
        tar --no-same-owner --zstd -xvf *python-numpy*.tar.zst --wildcards --no-anchored 'lib/python*/site-packages/*'
        mv mingw64/lib/python*/site-packages/* "%{python-site-packages-dir}"
        tar --no-same-owner --zstd -xvf *openblas*.tar.zst --wildcards --no-anchored 'bin/*'
        mv mingw64/bin/libopenblas.dll "%{install-root}%{bindir}"
        tar --no-same-owner --zstd -xvf *libgfortran*.tar.zst --wildcards --no-anchored 'bin/*'
        mv mingw64/bin/libgfortran*.dll "%{install-root}%{bindir}"
        find "%{python-site-packages-dir}" -regex '^.*\(__pycache__\|\.py[co]\)$' -delete
      else
        exit 1
      fi

build-depends:
- sdk.bst
- zstd.bst
- freedesktop-sdk.bst:components/libarchive.bst
