kind: script

description: |

    Remove unwanted files.

config:
  root-read-only: False
  commands:
    - rm -rf %{prefix}/compilation
    - rm -rf %{prefix}/include
    - rm -rf %{prefix}/lib/cmake
    - rm -rf %{prefix}/sbin
    - rm -rf %{prefix}/lib/*.a %{prefix}/lib/*.la %{prefix}/lib/*.prl
    - rm -rf %{prefix}/man
    - rm -rf %{prefix}/examples
    - rm -rf %{prefix}/doc*
    - rm -rf %{prefix}/mkspecs
    - cd %{prefix}/share && ls -I squey -I qt6 -I pocl -I doc -I applications -I icons -I app-info | xargs rm -rf
    - cd %{prefix}/share/doc && ls -I squey | xargs rm -rf
    - cd %{prefix}/bin && ls -I run_cmd.sh -I squey_launcher.sh -I squey -I squey-crashreport -I squey-logstats -I pip3 -I pip3.10 -I pwgen -I tshark -I lbzip2 -I lbunzip2 -I pigz -I unpigz |xargs rm -rf
    
    - cp -r %{prefix} %{install-root}

build-depends:
- base.bst
- squey.bst
