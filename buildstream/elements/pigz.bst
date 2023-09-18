kind: autotools
description: |

    A parallel implementation of gzip for modern multi-processor, multi-core machines

sources:
- kind: tar

  url: zlib:pigz/pigz-2.8.tar.gz
  ref: eb872b4f0e1f0ebe59c9f7bd8c506c4204893ba6a8492de31df416f0d5170fd0 

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
