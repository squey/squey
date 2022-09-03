kind: junction
sources:
- kind: git_tag
  url: https://gitlab.com/freedesktop-sdk/freedesktop-sdk
  track: 'freedesktop-sdk-22.08.0'
  track-tags: true
  submodules:
    utils/buildstream-abi-checker:
      url: https://gitlab.com/freedesktop-sdk/buildstream-abi-checker.git
  ref: freedesktop-sdk-22.08.0-0-gab971d8eb94802d2f94ea3b75aaeaef133233837
