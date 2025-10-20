kind: manual
description: |

    Info-ZIP portable compression/archiver utilities (Zip, UnZip, WiZ, etc.)

sources:
- kind: remote
  url: sourceforge:project/gnuwin32/zip/3.0/zip-3.0-bin.zip
  ref: 291c7404c728e0714fc95148b526ce471bdc09c45a65a4e0c51bff4a00825ad2
- kind: remote
  url: sourceforge:project/gnuwin32/zip/3.0/zip-3.0-dep.zip
  ref: 2e60c455e06664ac633507d2d4ccda6a09b78f59a9b70e82201d2c5e8b9305d2
- kind: remote
  url: sourceforge:project/gnuwin32/unzip/5.51-1/unzip-5.51-1-bin.zip
  ref: 3580e7908dab5136da7d84df6bee81499c6591ee50b40db80721ed3a2f43247b

(@): buildstream/elements/base/target.yml

config:
  (?):
  - (target_triple == "x86_64-w64-mingw32"):
      install-commands:
      - |
        7zzs x zip-3.0-bin.zip bin
        7zzs x zip-3.0-dep.zip bin
        7zzs x unzip-5.51-1-bin.zip bin
        install -D bin/zip.exe bin/unzip.exe bin/bzip2.dll bin/funzip.exe -t %{install-root}%{bindir}

build-depends:
- sdk.bst
- 7zip_host.bst
