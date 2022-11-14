kind: distutils

depends:
- base.bst
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/python3-setuptools.bst

config:
  install-commands:
    (>):
    - |
      rm "%{install-root}%{bindir}/pip"

sources:
- kind: git_tag
  url: https://github.com/pypa/pip.git
  track: master
  ref: 22.3.1-0-g1463081f10de6bfad81afe0d68272e7c3bedbadf
  exclude:
  - '*b?'
  - '*rc?'
  - -*a?'
