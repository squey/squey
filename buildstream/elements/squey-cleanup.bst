kind: script

description: |

    Remove unwanted files.

(@): buildstream/elements/base/target.bst

config:
  root-read-only: False
  commands:
    - |
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
      [ cd %{prefix}/share/qt6 ] && ls -I resources | xargs rm -rf
      cd %{prefix}/share/doc && ls -I squey | xargs rm -rf
      cd %{prefix}/bin && ls -I run_cmd.sh -I squey_launcher.sh -I squey -I squey_sh -I squey-crashreport -I squey-logstats -I clang -I clang++ -I ld -I pip3 -I pip3.12 -I pwgen -I tshark -I lbzip2 -I lbunzip2 -I pigz -I unpigz -I xz -I unxz |xargs rm -rf
      cp -r %{prefix} %{install-root}

build-depends:
- base.bst
- squey.bst
