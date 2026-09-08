kind: pyproject
description: |

    Meson® is a project to create the best possible next-generation build system.

    This is the numpy fork that contains the required "features" meson module

sources:
- kind: git_repo
  url: github:numpy/meson.git
  track: '*.*.*'
  exclude:
  - '*rc*'
  ref: 983fca5b39b62060c733a58fd8ecd276591bdf12

(@): buildstream/elements/base/target.yml

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/python3-setuptools.bst
- freedesktop-sdk.bst:components/python3-build.bst
- freedesktop-sdk.bst:components/python3-wheel.bst
- freedesktop-sdk.bst:components/python3-installer.bst

runtime-depends:
- freedesktop-sdk.bst:components/python3.bst
