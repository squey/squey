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
  ref: 6f88e485f27bb0a41d31638f0c55055362e0b1ac

build-depends:
- base.bst
- freedesktop-sdk.bst:components/python3-setuptools.bst
- freedesktop-sdk.bst:components/python3-build.bst
- freedesktop-sdk.bst:components/python3-wheel.bst
- freedesktop-sdk.bst:components/python3-installer.bst

depends:
- freedesktop-sdk.bst:components/python3.bst