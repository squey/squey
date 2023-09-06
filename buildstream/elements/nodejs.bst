kind: autotools
description: |

    As an asynchronous event-driven JavaScript runtime, Node.js is designed to build scalable network applications.

sources:
- kind: tar

  url: https://nodejs.org/dist/v20.4.0/node-v20.4.0.tar.xz
  ref: 09bd0b73c526b63c029d5ddfd885d10962e7ad87c975b94583c1f8ce90ee5348

config:
  configure-commands:
  - ./configure --prefix=%{prefix} --ninja

depends:
- base.bst
- freedesktop-sdk.bst:components/python3.bst