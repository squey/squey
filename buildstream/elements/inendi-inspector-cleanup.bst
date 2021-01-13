kind: script

description: |

    Remove unwanted files.

config:
  root-read-only: False
  commands:
    - rm -rf %{prefix}/include
    - rm -rf %{prefix}/lib/cmake
    - rm -rf %{prefix}/sbin
    - rm -rf %{prefix}/lib/*.a %{prefix}/lib/*.la %{prefix}/lib/*.prl
    - rm -rf %{prefix}/man
    - rm -rf %{prefix}/examples
    - rm -rf %{prefix}/doc*
    - rm -rf %{prefix}/mkspecs
    - cd %{prefix}/share && ls -I inendi -I qt5 -I pocl -I doc -I applications -I icons | xargs rm -rf
    - cd %{prefix}/share/doc && ls -I inendi-inspector | xargs rm -rf
    - cd %{prefix}/bin && ls -I run_cmd.sh -I inspector_launcher.sh -I inendi-inspector -I inendi-crashreport -I inendi-logstats -I inendi-echoid -I inendi-lsinit -I pip3 -I pip3.8 -I pwgen -I tshark -I lbzip2 -I lbunzip2 -I pigz -I unpigz |xargs rm -rf
    
    - cp -r %{prefix} %{install-root}

build-depends:
- base.bst
- inendi-inspector.bst
