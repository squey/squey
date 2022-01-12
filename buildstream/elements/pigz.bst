kind: autotools
description: |

    A parallel implementation of gzip for modern multi-processor, multi-core machines

sources:
- kind: tar

  url: https://zlib.net/pigz/pigz-2.6.tar.gz
  ref: 2eed7b0d7449d1d70903f2a62cd6005d262eb3a8c9e98687bc8cbb5809db2a7d

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
