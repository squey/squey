kind: autotools
description: |

    LCOV is a graphical front-end for GCC's coverage testing tool gcov.
    It collects gcov data for multiple source files and creates HTML pages containing the source code
    annotated with coverage information.
    It also adds overview pages for easy navigation within the file structure.
    LCOV supports statement, function and branch coverage measurement.
    
sources:
- kind: tar  
  url: https://github.com/linux-test-project/lcov/archive/refs/tags/v1.16.tar.gz
  ref: 02eece2802697c079ff51ce056b0a6691020f21082c46ccfe6604bbfd06d7937

config:
  configure-commands:
  - true

depends:
- base.bst