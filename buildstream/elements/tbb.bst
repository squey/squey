kind: autotools
description: |

    Intel® Threading Building Blocks (Intel® TBB) is a widely used C++ library for shared memory
    parallel programming and heterogeneous computing (intra-node distributed memory programming).

sources:
- kind: tar
  #url: https://github.com/oneapi-src/oneTBB/archive/2020_U1.tar.gz
  #ref: d80ca22c224ab7ef913dfae72c23fc1434b6aa46bfd472916d8c874c90881f5e
  
  url: https://github.com/01org/tbb/archive/2018_U5.tar.gz
  ref: b8dbab5aea2b70cf07844f86fa413e549e099aa3205b6a04059ca92ead93a372

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
