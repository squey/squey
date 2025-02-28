kind: manual

sources:
- kind: local
  path: buildstream/files/msi_package
  directory: buildstream/files/msi_package
- kind: local
  path: ./src/gui-qt/src/resources/squey.ico
- kind: local
  path: VERSION.txt

(@):
    - buildstream/elements/base/target.bst
    - buildstream/elements/msi_package/config.yml

environment:
  LD_LIBRARY_PATH: "%{host-prefix}/lib"

variables:
    wxs-project: "squey.wxs"
    wxs-files: "squey-files.wxs"
    package-name: "squey"
    package-root: "msi_package_root"

config:
  configure-commands:
  - |
    SOURCE_DIR=$(pwd)
    mkdir -p "%{package-root}/Squey/plugins"
    cp %{bindir}/*.dll "%{package-root}/Squey"

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
    cd "%{package-root}/Squey"
    extract_local_deps "pocl.dll"
    echo "LLVM/Clang deps: " $deps_list
    rm -rf libLLVM*.dll libclang*.dll
    cd "%{bindir}"
    cp $deps_list "${SOURCE_DIR}/%{package-root}/Squey"
    cd "${SOURCE_DIR}"

    cp -r --backup=numbered \
      /usr/%{host}/lib/{libstdc++-6.dll,libgcc_s_seh-1.dll,libgomp-1.dll} \
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
      "%{package-root}/Squey"
    cp -r %{libdir}/squey/plugins/* "%{package-root}/Squey/plugins"
    cp %{bindir}/pocl/* "%{package-root}/Squey/pocl"
    mkdir -p "%{package-root}/Squey/python"
    cp -r %{libdir}/python*/ "%{package-root}/Squey/python"
    rm -rf %{package-root}/Squey/libjemalloc.* %{package-root}/Squey/jemalloc.*

  build-commands:
  - |
    VERSION=$(cat VERSION.txt)
    PACKAGE_NAME="%{package-name}_${VERSION}.msi"
    
    sed -e "s|{{version}}|${VERSION}|" -e "s|{{package_root}}|%{package-root}|" "%{build-root}/buildstream/files/msi_package/%{wxs-project}.j2" > %{wxs-project}
    find "%{package-root}" | wixl-heat --prefix "%{package-root}/" --directory-ref LocalAppDataFolder --component-group CG.squey --var var.SourceDir > "%{wxs-files}"
    wixl -v -D SourceDir="%{package-root}" -o "%{install-root}/${PACKAGE_NAME}" "%{wxs-project}" "%{wxs-files}"
    msiextract -l "%{install-root}/${PACKAGE_NAME}"
    
build-depends:
- squey-cleanup.bst
- msi_package/msitools.bst
- sdk.bst