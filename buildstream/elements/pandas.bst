kind: manual
description: |

    pandas: powerful Python data analysis toolkit

sources:
- kind: tar
  url: https://github.com/pandas-dev/pandas/archive/v1.0.5.tar.gz
  ref: f9aec4583b0338f18865576b1356182cb425bd7dd9d6faf918c6c4ab45d7fffe
  
environment:
  PYTHONPATH: "%{prefix}/lib/python3.7/site-packages"
  
config:
  install-commands:
    - python3 setup.py build -j `nproc` install --root=%{install-root} --prefix=%{prefix}

depends:
- base.bst
- freedesktop-sdk.bst:components/python3.bst
- numpy.bst

build-depends:
- freedesktop-sdk.bst:components/python3-setuptools.bst
- freedesktop-sdk.bst:components/cython.bst
