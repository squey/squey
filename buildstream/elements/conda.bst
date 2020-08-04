kind: manual
description: |

    Conda is a cross-platform, language-agnostic binary package manager.
    It is the package manager used by Anaconda installations, but it may be used for other systems as well.
    Conda makes environments first-class citizens, making it easy to create independent environments even for C libraries.
    Conda is written entirely in Python, and is BSD licensed open source.

sources:
- kind: git
  url: https://github.com/conda/conda.git
  ref: 42a06816356a905cce90564ea4663e2787832467
  
config:
  install-commands:
    - python3 setup.py install --root=%{install-root} --prefix=%{prefix}

depends:
- base.bst
- freedesktop-sdk.bst:components/python3.bst

build-depends:
- freedesktop-sdk.bst:components/python3-setuptools.bst
- freedesktop-sdk.bst:components/git.bst
 
