kind: manual
description: |

    Microsoft MSIX SDK

sources:
- kind: git
  url: github:microsoft/msix-packaging
  ref: efeb9dad695a200c2beaddcba54a52c8320bd135
  checkout-submodules: True

config:
  configure-commands:
    - |
      sed -e 's|^cmake -D|cmake -DCMAKE_INSTALL_PREFIX:PATH=%{prefix} -DCMAKE_INSTALL_LIBDIR:PATH=%{lib} -D|' -e 's|^make|make\nDESTDIR=%{install-root} make install|' -i makelinux.sh

      # Force using specific ICU version
      echo -e "\nset(CMAKE_PREFIX_PATH %{prefix})" >> cmake/linux.cmake
      echo -e "set(ICU_INCLUDE_DIR \"%{includedir}/unicode\")" >> cmake/linux.cmake
      echo -e "\ninclude_directories(BEFORE %{includedir})" > lib/xerces/cmake/XercesICU.cmake
      echo -e "set(CMAKE_PREFIX_PATH %{prefix})" >> lib/xerces/cmake/XercesICU.cmake
      echo -e "set(ICU_ROOT \"%{prefix}\")" >>  lib/xerces/cmake/XercesICU.cmake
      echo -e "set(ICU_INCLUDE_DIR \"%{includedir}/unicode\")" >> lib/xerces/cmake/XercesICU.cmake
      echo -e "find_package(ICU COMPONENTS uc data)" >> lib/xerces/cmake/XercesICU.cmake
      echo -e "\ninclude_directories(BEFORE %{includedir})" >> src/msix/CMakeLists.txt

      ./makelinux.sh --pack
      mkdir -p "%{install-root}%{bindir}" "%{install-root}%{libdir}"
      cp -r .vs/bin/* "%{install-root}%{bindir}"
      cp -r .vs/lib/* "%{install-root}%{libdir}"

build-depends:
- clang_host.bst

depends:
- base.bst
- msix_package/icu.bst
