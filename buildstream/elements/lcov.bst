kind: autotools
description: |

    LCOV is a graphical front-end for GCC's coverage testing tool gcov.
    It collects gcov data for multiple source files and creates HTML pages containing the source code
    annotated with coverage information.
    It also adds overview pages for easy navigation within the file structure.
    LCOV supports statement, function and branch coverage measurement.

sources:
- kind: tar
  url: github:linux-test-project/lcov/archive/refs/tags/v2.3.2.tar.gz
  ref: c77dbd4b8919c13ca9450a20482d40b386fee6bec6cef33be1790bf3d1bb7d3e

config:
  configure-commands:
  - true

depends:
- base.bst
