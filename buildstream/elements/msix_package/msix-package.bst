kind: manual
description: |

    All dependencies to create Squey MSIX package (application file layout and build tools)

    This element is used by "make-msix-package.sh" script within "bst shell" to avoid caching the artifact

sources:
- kind: local
  path: buildstream/files/msix_package
  directory: buildstream/files/msix_package
- kind: local
  path: buildstream/elements/msix_package/config.yml
  directory: buildstream/files/msix_package
- kind: local
  path: VERSION.txt

(@):
    - buildstream/elements/base/target.bst
    - buildstream/elements/msix_package/config.yml

environment:
  LD_LIBRARY_PATH: "%{host-prefix}/lib"
    
build-depends:
- msix_package/msix-package-layout.bst
- squey-cleanup.bst
- msix_package/msix-packaging.bst
- msix_package/yq.bst
- 7zip_host.bst
- sdk.bst

runtime-depends:
- freedesktop-sdk.bst:bootstrap/bash.bst