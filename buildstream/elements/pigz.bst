kind: autotools
description: |

    A parallel implementation of gzip for modern multi-processor, multi-core machines

sources:
- kind: tar
  url: zlib:pigz/pigz-2.8.tar.gz
  ref: eb872b4f0e1f0ebe59c9f7bd8c506c4204893ba6a8492de31df416f0d5170fd0

(@): buildstream/elements/base/target.bst


config:
  configure-commands:
  - |
    cat <<EOF >> "Makefile"
    install:
    	mkdir -p \$(DESTDIR)/bin
    	install -m 0755 pigz \$(DESTDIR)/bin
    	install -m 0755 unpigz \$(DESTDIR)/bin
    EOF
    if [ "%{target_triple}" = "x86_64-apple-darwin" ] ; then sed 's/CC=gcc/CC=o64-clang/' -i "Makefile"; fi
    if [ "%{target_triple}" = "aarch-apple-darwin" ] ; then sed 's/CC=gcc/CC=oa64-clang/' -i "Makefile"; fi
  install-commands:
  - make -j 1 install DESTDIR="%{install-root}%{prefix}"

build-depends:
- sdk.bst
- zlib.bst
- 7zip_host.bst
