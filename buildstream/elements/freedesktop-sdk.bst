kind: junction

sources:
- kind: git_tag
  url: https://gitlab.com/freedesktop-sdk/freedesktop-sdk
  track: '20.08'
  track-tags: true
  ref: freedesktop-sdk-20.08.0-0-g5931bbdc1a1d3da5f3218a718ff1b4f28d6b114f
  submodules:
    utils/buildstream-abi-checker:
      url: https://gitlab.com/freedesktop-sdk/buildstream-abi-checker.git
