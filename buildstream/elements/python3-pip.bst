kind: pyproject

# pip dropped setup.py in favour of a PEP 517 build (flit-core backend) as of
# 26.x, so this is no longer a 'setuptools' element.

build-depends:
- base.bst
- freedesktop-sdk.bst:components/python3-flit-core.bst
- freedesktop-sdk.bst:components/python3-build.bst
- freedesktop-sdk.bst:components/python3-wheel.bst
- freedesktop-sdk.bst:components/python3-installer.bst

runtime-depends:
- freedesktop-sdk.bst:components/python3.bst

config:
  # The pyproject plugin runs 'installer' without a prefix, which lands the
  # files under the sandbox python's own /usr rather than the app prefix, so
  # the install step is replaced rather than appended to.
  install-commands:
    (=):
    - |
      %{python} -minstaller %{dist-dir}/*.whl --destdir "%{install-root}" --prefix "%{prefix}"
    - |
      rm "%{install-root}%{bindir}/pip"

sources:
- kind: git_tag
  url: github:pypa/pip.git
  track: master
  ref: 26.2.1-0-g634a6ec1a5d9dcc2433571cdb2f4c58a4bb29caf
  exclude:
  - '*b?'
  - '*rc?'
  - -*a?'
