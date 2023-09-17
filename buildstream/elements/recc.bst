kind: cmake

depends:
- base.bst
- buildbox-common.bst
- freedesktop-sdk.bst:components/protobuf.bst
- freedesktop-sdk.bst:components/grpc.bst
- freedesktop-sdk.bst:components/glog.bst
- freedesktop-sdk.bst:components/pkg-config.bst

sources:
- kind: git
  url: https://gitlab.com/BuildGrid/recc.git
  ref: 1f48bc5c11d07a4b03cab1bf475e2ed1feff1a7d

- kind: patch
  path: buildstream/patches/recc/recc.patch
  strip-level: 1

variables:
  cmake-local: |
    -DBUILD_TESTING=OFF