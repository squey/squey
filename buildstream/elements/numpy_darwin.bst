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

(@): buildstream/elements/base/target.bst

variables:
  python-site-packages-dir: "%{install-root}%{libdir}/python3.12/site-packages"

config:
  install-commands:
    - |
      mkdir -p "%{python-site-packages-dir}"
      unzip *.whl 'numpy/*' -d "%{python-site-packages-dir}"
    
build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/unzip.bst
