kind: distutils

sources:
- kind: tar
  url: https://github.com/gsnedders/python-webencodings/archive/refs/tags/v0.5.1.tar.gz
  ref: 082367f568a7812aa5f6922ffe3d9d027cd83829dc32bcaac4c874eeed618000

depends:
- base.bst
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/python3-setuptools.bst