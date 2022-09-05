kind: autotools
description: |

    As an asynchronous event-driven JavaScript runtime, Node.js is designed to build scalable network applications.

sources:
- kind: tar

  url: https://nodejs.org/dist/v18.8.0/node-v18.8.0.tar.xz
  ref: 2b5d9825d05ede6614f1668a8d97d774fe92ebc81088ec5fdf58184dce3c86b9

config:
  configure-commands:
  - ./configure --prefix=%{prefix} --ninja

depends:
- base.bst