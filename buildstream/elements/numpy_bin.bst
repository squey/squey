kind: manual
description: |

    NumPy is the fundamental package needed for scientific computing with Python.

(?):
  - target_triple == "x86_64-apple-darwin":
      sources:
        - kind: remote
          url: pythonhosted:packages/a7/45/7f9244cd792e163b334e3a7f02dff1239d2890b6f37ebf9e82cbe17debc0/numpy-2.1.3-cp313-cp313t-macosx_10_13_x86_64.whl
          ref: 996bb9399059c5b82f76b53ff8bb686069c05acc94656bb259b1d63d04a9506f
  - target_triple == "aarch64-apple-darwin":
      sources:
        - kind: remote
          url: pythonhosted:packages/45/e1/210b2d8b31ce9119145433e6ea78046e30771de3fe353f313b2778142f34/numpy-2.1.3-cp313-cp313-macosx_11_0_arm64.whl
          ref: f653490b33e9c3a4c1c01d41bc2aef08f9475af51146e4a7710c450cf9761598
  - target_triple == "x86_64-w64-mingw32":
      sources:
        - kind: remote
          url: msys2:mingw/mingw64/mingw-w64-x86_64-python-numpy-2.2.3-1-any.pkg.tar.zst
          ref: 55a1b75c198a845a7fe17d6090f303fcd992016775f3817794a2d8d6cedecd29
        - kind: remote
          url: msys2:mingw/mingw64/mingw-w64-x86_64-openblas-0.3.29-1-any.pkg.tar.zst
          ref: e0c22584647202589901dd08715b814fe9529830848742fe19682dd34cabf245
        - kind: remote
          url: msys2:mingw/mingw64/mingw-w64-x86_64-gcc-libgfortran-14.2.0-3-any.pkg.tar.zst
          ref: 6f5d0967234ccc25f18556308293be70d3a63ca8c285033e561593a7d754fe24

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
- freedesktop-sdk.bst:components/unzip.bst
