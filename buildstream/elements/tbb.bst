kind: autotools
description: |

    Intel® Threading Building Blocks (Intel® TBB) is a widely used C++ library for shared memory
    parallel programming and heterogeneous computing (intra-node distributed memory programming).

sources:
- kind: tar  
  url: github:oneapi-src/oneTBB/archive/v2020.3.tar.gz
  ref: ebc4f6aa47972daed1f7bf71d100ae5bf6931c2e3144cf299c8cc7d041dca2f3

- kind: patch
  path: buildstream/patches/tbb/task.patch
  strip-level: 1

config:
  configure-commands:
  - |
    cat <<EOF >> "Makefile"
    	
    install:
    	mkdir -p \$(DESTDIR)/include \$(DESTDIR)/lib
    	cp -r include/* \$(DESTDIR)/include
    	cp \$(work_dir)_release/*.so* \$(DESTDIR)/lib
    	rm -rf \$(DESTDIR)/lib/debug
    EOF
  install-commands:
  - make -j 1 install DESTDIR="%{install-root}%{prefix}"
  
depends:
- base.bst
