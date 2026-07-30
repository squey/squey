kind: script

description: |

    Remove unwanted files.

(@): buildstream/elements/base/target.yml

environment:
  # crashpad_handler and squey-crashreport are needed on the three platforms:
  # the handler captures the minidump out of process, the reporter asks for
  # consent and uploads it.
  common_bin_files: "squey tshark lbzip2 funzip pigz zstd xz pip3 pip3.13 crashpad_handler squey-crashreport"
  linux_bin_files: "ld 7zzs run_cmd.sh squey_sh squey_launcher.sh squey-logstats"
  darwin_bin_files: "lld ld64.lld 7zz"
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
      # The installed testsuite is only needed by the macOS and Windows packaging,
      # which turns %{prefix}/tests into testsuite.zip to be run on the target
      # platform. Linux runs its tests from the build tree, so keep them out of
      # the published package.
      if [ "%{target-platform}" = "linux" ]; then
        rm -rf %{prefix}/tests %{prefix}/lib/debug/tests
      fi
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
        # The executables used to be stripped at link time by a '-s' in
        # CMakeCompilers.txt, which left no symbols to dump. They are stripped
        # here instead, after build.sh has sent the symbols to the crash server.
        find %{libdir} %{bindir} \( -name "*.dll" -o -name "*.exe" \) ! -name "cygwin1.dll" | xargs %{host}-strip -s # cygwin1.dll doesn't support symbols stripping
      fi

      cp -r %{prefix} %{install-root}

build-depends:
- base.bst
- squey.bst
