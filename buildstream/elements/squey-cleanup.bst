kind: script

description: |

    Remove unwanted files.

(@): buildstream/elements/base/target.bst

environment:
  common_bin_files: "squey tshark 7zz lbzip2 funzip pigz zstd xz pip3 pip3.12"
  linux_bin_files: "ld run_cmd.sh squey_sh squey_launcher.sh squey-crashreport squey-logstats"
  darwin_bin_files: "lld ld64.lld"
  windows_bin_files: "ld 7z pbzip2 python pocl \"*.dll\""

config:
  root-read-only: False
  commands:
    - |
      set -e
      set -x
      rm -rf %{prefix}/compilation_build
      rm -rf %{prefix}/junit.xml
      rm -rf %{prefix}/code_coverage_report
      rm -rf %{includedir}
      rm -rf %{prefix}/lib/cmake
      rm -rf %{prefix}/sbin
      find %{prefix}/lib \( -name "*.a" -o -name "*.la" -o -name "*.prl" \) ! \( -name "libgcc.a" -o -name "libc_nonshared.a" \) -exec rm -rf {} \;
      rm -rf %{prefix}/man
      rm -rf %{prefix}/examples
      rm -rf %{prefix}/doc*
      rm -rf %{prefix}/mkspecs
      cd %{prefix}/share && ls -I squey -I qt6 -I pocl -I doc -I applications -I icons -I app-info | xargs rm -rf
      [ -d %{prefix}/share/qt6 ] && cd %{prefix}/share/qt6 && ls -I resources | xargs rm -rf
      cd %{prefix}/share/doc && ls -I squey | xargs rm -rf

      target_platform_files="%{target-platform}_bin_files"
      all_files="$common_bin_files ${!target_platform_files}"
      cmd=$(echo "$all_files" | sed 's/\([^ ]*\)/-I \1 -I \1.exe/g')
      cmd="ls $cmd | xargs rm -rf"
      cd %{prefix}/bin && eval $cmd

      if [ "%{target-platform}" = "windows" ] ; then
        find %{libdir} %{bindir} -name "*.dll" ! -name "cygwin1.dll" | xargs %{host}-strip -s # cygwin1.dll doesn't support symbols stripping
      fi

      cp -r %{prefix} %{install-root}

build-depends:
- base.bst
- squey.bst
