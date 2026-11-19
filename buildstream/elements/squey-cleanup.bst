kind: script

description: |

    Remove unwanted files.

(@): buildstream/elements/base/target.yml

environment:
  # What Squey installs in %{bindir} is kept, as install_manifest.txt lists it
  # (see squey.bst). These are the tools of its dependencies to keep beside it:
  # crashpad_handler captures the minidump out of process, for squey-crashreport
  # to ask for consent and upload it.
  common_bin_files: "tshark lbzip2 funzip pigz zstd xz pip3 pip%{python-version} crashpad_handler"
  linux_bin_files: "ld 7zzs"
  darwin_bin_files: "lld ld64.lld 7zz"
  windows_bin_files: "ld 7z python pocl \"*.dll\""

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

      own_bin_files=$(sed -n 's|^%{bindir}/\([^/]*\)$|\1|p' %{prefix}/install_manifest.txt | paste -sd ' ')
      rm %{prefix}/install_manifest.txt
      target_platform_files="%{target-platform}_bin_files"
      all_files="$own_bin_files $common_bin_files ${!target_platform_files}"
      cmd=$(echo "$all_files" | sed 's/\([^ ]*\)/-I \1 -I \1.exe/g')
      cmd="ls $cmd | xargs rm -rf"
      cd %{prefix}/bin && eval $cmd

      if [ "%{target-platform}" = "windows" ] ; then
        # The executables used to be stripped at link time by a '-s' in
        # CMakeCompilers.txt, which left no symbols to dump. They are stripped
        # here instead, after build.sh has sent the symbols to the crash server.
        # %{prefix}/tests is stripped too: make-msix-package.sh zips it whole into
        # testsuite.zip, and the DWARF of the test binaries was 96% of that archive.
        # PE keeps its debug information inside the binary, unlike Mach-O, which
        # leaves it in the object files -- hence no macOS counterpart to this.
        # A build that skips the testsuite has no such directory.
        strip_dirs="%{libdir} %{bindir}"
        [ -d %{prefix}/tests ] && strip_dirs="$strip_dirs %{prefix}/tests"
        find $strip_dirs \( -name "*.dll" -o -name "*.exe" \) | xargs %{host}-strip -s
      fi

      cp -r %{prefix} %{install-root}

build-depends:
- base.bst
- squey.bst
