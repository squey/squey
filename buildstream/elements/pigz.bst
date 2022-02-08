kind: autotools
description: |

    A parallel implementation of gzip for modern multi-processor, multi-core machines

sources:
- kind: tar

  url: https://zlib.net/pigz/pigz-2.7.tar.gz 
  ref: b4c9e60344a08d5db37ca7ad00a5b2c76ccb9556354b722d56d55ca7e8b1c707 

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
