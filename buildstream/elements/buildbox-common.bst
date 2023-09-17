kind: cmake

sources:
- kind: git
  url: https://gitlab.com/BuildGrid/buildbox/buildbox-common.git
  ref: 6e70ed24cfc654031308f7b7b99ed68f649562cd
  
depends:
- base.bst
- freedesktop-sdk.bst:components/protobuf.bst
- freedesktop-sdk.bst:components/grpc.bst
- freedesktop-sdk.bst:components/glog.bst
- freedesktop-sdk.bst:components/pkg-config.bst

variables:
  cmake-local: |
    -DBUILD_TESTING=OFF
