kind: manual
description: |

    Squey MSIX package files layout

sources:
- kind: local
  path: buildstream/files/msix_package/Square44x44Logo.png
- kind: local
  path: buildstream/files/msix_package/Square150x150Logo.png
- kind: local
  path: buildstream/files/msix_package/resources.pri
- kind: local
  path: ./src/gui-qt/src/resources/squey.ico
- kind: local
  path: ./buildstream/files/org.squey.Squey.png
- kind: local
  path: VERSION.txt

(@):
    - buildstream/elements/base/target.bst
    - buildstream/elements/msix_package/config.yml

variables:
  install-package-root: "%{install-root}/%{package-root}"

config:
  install-commands:
  - |
    SOURCE_DIR=$(pwd)
    mkdir -p "%{install-package-root}/plugins"
    cp %{bindir}/*.dll "%{install-package-root}"

    # Remove unused LLVM/Clang libraries
    extract_local_deps() {
      local file="$1"
      local deps=$(%{host}-objdump -p "$file" 2>/dev/null | grep 'DLL Name' | awk '{print $3}')
      for dep in $deps; do
          if [[ -f "./$dep" && ! " ${seen[*]} " =~ " $dep " ]]; then
              seen+=("$dep")
              deps_list+=" $dep"
              extract_local_deps "./$dep"
          fi
      done
    }
    deps_list=""
    declare -a seen
    cd "%{install-package-root}"
    extract_local_deps "pocl.dll"
    echo "LLVM/Clang deps: " $deps_list
    rm -rf libLLVM*.dll libclang*.dll
    cd "%{bindir}"
    cp $deps_list "%{install-package-root}"
    cd "${SOURCE_DIR}"

    cp -r --backup=numbered \
      /usr/%{host}/lib/{libstdc++-6.dll,libgcc_s_seh-1.dll,libgomp-1.dll,libquadmath-0.dll} \
      /usr/%{host}/lib/{dllcrt2.o,crtbegin.o,crtend.o} \
      /usr/%{host}/lib/{libmingw32.a,libgcc_s.a,libmoldname.a,libmingwex.a,libmsvcrt.a,libadvapi32.a,libshell32.a,libuser32.a,libkernel32.a} \
      /usr/lib/gcc/%{host}/*/libgcc.a \
      /usr/%{host}/bin/libwinpthread-1.dll \
      %{bindir}/*.exe \
      %{libdir}/*.dll \
      %{libdir}/qt6/plugins/platforms \
      %{libdir}/qt6/plugins/sqldrivers \
      %{prefix}/share/pocl \
      %{prefix}/share/squey/squey/{pvconfig.ini,COPYING,CHANGELOG} \
      squey.ico \
      org.squey.Squey.png \
      Square44x44Logo.png \
      Square150x150Logo.png \
      resources.pri \
      "%{install-package-root}"
    cp "Square44x44Logo.png" "%{install-package-root}/Square44x44Logo.targetsize-44_altform-unplated.png"
    cp "Square150x150Logo.png" "%{install-package-root}/Square150x150Logo.targetsize-150_altform-unplated.png"
    cp -r %{libdir}/squey/plugins/* "%{install-package-root}/plugins"
    cp %{bindir}/pocl/* "%{install-package-root}/pocl"
    mkdir -p "%{install-package-root}/python"
    cp -r %{libdir}/python*/* "%{install-package-root}/python"
    cd "%{bindir}/tshark" && 7zz a "%{install-package-root}/tshark.zip" .
    rm -rf %{install-package-root}/libjemalloc.* %{install-package-root}/jemalloc.*

build-depends:
- squey-cleanup.bst
- sdk.bst
- 7zip_host.bst
