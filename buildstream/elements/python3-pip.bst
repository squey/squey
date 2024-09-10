kind: setuptools

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
  url: github:pypa/pip.git
  track: master
  ref: 24.0-0-gef78c129b1a966dbbbdb8ebfffc43723e89110d1
  exclude:
  - '*b?'
  - '*rc?'
  - -*a?'
