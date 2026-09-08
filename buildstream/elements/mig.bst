kind: autotools
description: |

    MIG, the Mach Interface Generator, runs on the build host and generates the
    Mach RPC stubs the macOS Crashpad handler needs (exc, mach_exc, notify,
    child_port).

    Apple only ships mig with an Xcode build system, so this is the fork of the
    Crashpad maintainer, which adds an Autotools build usable outside of macOS.
    The 'mig' driver script looks migcom up next to itself, so both have to land
    in the same bindir.

sources:
- kind: git
  url: github:markmentovai/bootstrap_cmds.git
  track: cross_platform
  ref: f337e600741638896fb6056ce87c6479f74b89a3

variables:
  # migcom declares 'extern char *strbool(boolean_t bool)': 'bool' became a
  # keyword in C23, which is what recent GCC defaults to.
  conf-local: ""

environment:
  CFLAGS: "-std=gnu17 -O2"

config:
  configure-commands:
  - autoreconf -fi
  - "%{configure}"

build-depends:
- base.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
