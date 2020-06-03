kind: autotools
description: |

    Boost provides free peer-reviewed portable C++ source libraries.

    We emphasize libraries that work well with the C++ Standard Library. Boost libraries are intended to be widely useful, and usable across a broad spectrum of applications. The Boost license encourages the use of Boost libraries for all users with minimal restrictions.

    We aim to establish "existing practice" and provide reference implementations so that Boost libraries are suitable for eventual standardization. Beginning with the ten Boost Libraries included in the Library Technical Report (TR1) and continuing with every release of the ISO standard for C++ since 2011, the C++ Standards Committee has continued to rely on Boost as a valuable source for additions to the Standard C++ Library.

sources:
- kind: tar

  #url: https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.bz2
  #ref: 59c9b274bc451cf91a9ba1dd2c7fdcaf5d60b1b3aa83f2c9fa143417cc660722
  url: https://netcologne.dl.sourceforge.net/project/boost/boost/1.59.0/boost_1_59_0.tar.bz2
  ref: 727a932322d94287b62abb1bd2d41723eec4356a7728909e38adb65ca25241ca
  
config:
  configure-commands:
  - |
    ./bootstrap.sh --prefix=%{prefix}
    cat <<EOF > "Makefile"
    LIB_TARGETS = --with-date_time --with-system --with-filesystem --with-thread --with-program_options --with-chrono
    all:
    	./b2 -j `nproc --ignore=1` \$(LIB_TARGETS)
    install:
    	./b2 install --prefix=\$(DESTDIR) \$(LIB_TARGETS)
    EOF
  install-commands:
  - make -j 1 install DESTDIR="%{install-root}%{prefix}"

depends:
- base.bst
