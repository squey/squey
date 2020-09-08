kind: autotools
description: |

    A parallel implementation of gzip for modern multi-processor, multi-core machines

sources:
- kind: tar

  url: https://zlib.net/pigz/pigz-2.4.tar.gz
  ref: a4f816222a7b4269bd232680590b579ccc72591f1bb5adafcd7208ca77e14f73

config:
  configure-commands:
  - |
    cat <<EOF >> "Makefile"
    	
    install:
    	mkdir -p \$(DESTDIR)/bin
    	install -m 0755 pigz \$(DESTDIR)/bin
    	install -m 0755 unpigz \$(DESTDIR)/bin
    EOF
  install-commands:
  - make -j 1 install DESTDIR="%{install-root}%{prefix}"


depends:
- base.bst
