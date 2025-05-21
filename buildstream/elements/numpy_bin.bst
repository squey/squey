kind: manual
description: |

    NumPy is the fundamental package needed for scientific computing with Python.

(?):
  - target_triple == "x86_64-apple-darwin":
      sources:
        - kind: remote
          url: pythonhosted:packages/8a/f0/385eb9970309643cbca4fc6eebc8bb16e560de129c91258dfaa18498da8b/numpy-2.1.3-cp312-cp312-macosx_10_13_x86_64.whl
          ref: f55ba01150f52b1027829b50d70ef1dafd9821ea82905b63936668403c3b471e
  - target_triple == "aarch64-apple-darwin":
      sources:
        - kind: remote
          url: pythonhosted:packages/54/4a/765b4607f0fecbb239638d610d04ec0a0ded9b4951c56dc68cef79026abf/numpy-2.1.3-cp312-cp312-macosx_11_0_arm64.whl
          ref: 13138eadd4f4da03074851a698ffa7e405f41a0845a6b1ad135b81596e4e9958
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

(@): buildstream/elements/base/target.bst

variables:
  python-site-packages-dir: "%{install-root}%{libdir}/python3.12/site-packages"

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
