kind: manual
description: |

    NumPy is the fundamental package needed for scientific computing with Python.

sources:
- kind: tar
  url: github:numpy/numpy/archive/refs/tags/v1.25.1.tar.gz
  ref: 77779aaf9bd7308b8cff3456144b22934a4aefa8e1b8f2bbadc18e21efcb5128
  
config:
  configure-commands:
  - |
    cat <<EOF >site.cfg
    [DEFAULT]
    library_dirs = /app/lib:/usr/local/lib64:/usr/local/lib:/usr/lib64:/usr/lib
    include_dirs = /app/include:/usr/local/include:/usr/include
    EOF
  install-commands:
    - python3 setup.py build --cpu-dispatch="max -avx512f -avx512cd -avx512_knl -avx512_knm -avx512_skx -avx512_clx -avx512_cnl -avx512_icl -avx512_spr" -j `nproc` install --root=%{install-root} --prefix=%{prefix}

depends:
- base.bst
- lapack.bst
- freedesktop-sdk.bst:components/python3.bst

build-depends:
- freedesktop-sdk.bst:components/python3-setuptools.bst
- freedesktop-sdk.bst:components/cython.bst
- freedesktop-sdk.bst:components/git.bst
 
