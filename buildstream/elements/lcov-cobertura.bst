kind: autotools
description: |

    This project does as the name implies: it converts code coverage report files in lcov format to
    Cobertura's XML report format so that CI servers like Jenkins can aggregate results and determine build stability etc.

sources:
- kind: tar
  url: github:eriwen/lcov-to-cobertura-xml/archive/refs/tags/2.0.2.tar.gz
  ref: bbbf0867ec7bf90d4a071097242edc88437a819d072a522cd79d8a44432861ca

config:
  configure-commands:
  - true
  build-commands:
  - true
  install-commands:
  - python setup.py install --root="%{install-root}"

depends:
- base.bst

build-depends:
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/python3-setuptools.bst
