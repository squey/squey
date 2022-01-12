kind: junction
sources:
- kind: git_tag
  url: https://gitlab.com/freedesktop-sdk/freedesktop-sdk
  track: 'freedesktop-sdk-20.08.17'
  track-tags: true
  submodules:
    utils/buildstream-abi-checker:
      url: https://gitlab.com/freedesktop-sdk/buildstream-abi-checker.git
  ref: freedesktop-sdk-20.08.17-0-g18906b5db4240c51a54e74ebab946e979ea0e357
