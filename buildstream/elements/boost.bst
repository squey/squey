kind: autotools
description: |

    Boost provides free peer-reviewed portable C++ source libraries.

    We emphasize libraries that work well with the C++ Standard Library. Boost libraries are intended to be widely useful, and usable across a broad spectrum of applications. The Boost license encourages the use of Boost libraries for all users with minimal restrictions.

    We aim to establish "existing practice" and provide reference implementations so that Boost libraries are suitable for eventual standardization. Beginning with the ten Boost Libraries included in the Library Technical Report (TR1) and continuing with every release of the ISO standard for C++ since 2011, the C++ Standards Committee has continued to rely on Boost as a valuable source for additions to the Standard C++ Library.

sources:
- kind: tar
  url: boost:release/1.92.0/source/boost_1_92_0.tar.bz2
  ref: 5c1d40cb8e19adbf740a4ec2da35b3e58f3f5804b1dce44deb53df72193cbc6c

(@): buildstream/elements/base/target.yml

config:
  configure-commands:
  - |
    ./bootstrap.sh --prefix=%{prefix}
    if [ "%{target-platform}" = "darwin" ] ; then
      # b2 hands the compiler -m64 whatever we ask of it: it reads the address
      # model off the build machine and clang-linux.jam turns that into a flag.
      # Targeting Mach-O, the clang driver rewrites -m64 into "-arch x86_64",
      # which outranks --target -- so the arm64 libraries came out x86_64 and
      # every boost symbol was missing when squey linked. Adding -arch arm64
      # does not override it either, it just yields a universal binary. Drop
      # the flag on its way to the compiler.
      mkdir -p wrappers
      printf '%s\n' \
        '#!/bin/bash' \
        'args=()' \
        'for arg in "$@" ; do [ "$arg" = "-m64" ] || args+=("$arg") ; done' \
        'exec %{cxx-compiler} "${args[@]}"' > wrappers/b2-cxx
      chmod +x wrappers/b2-cxx
      echo "using clang : osxcross : $PWD/wrappers/b2-cxx ;" > ~/user-config.jam
    elif [ "%{target_triple}" = "x86_64-w64-mingw32" ] ; then
      echo "using gcc : mingw : %{cxx-compiler} ;" > ~/user-config.jam
    fi
    cat <<EOF > "Makefile"
    LIB_TARGETS = --with-date_time --with-system --with-filesystem --with-thread --with-program_options --with-chrono
    all:
    	./b2 toolset=%{boost-toolset} target-os=%{boost-target-os} cxxflags="%{boost-cxx-flags}" linkflags="%{boost-link-flags}" -j `nproc --ignore=1` \$(LIB_TARGETS)
    install:
    	./b2 toolset=%{boost-toolset} target-os=%{boost-target-os} cxxflags="%{boost-cxx-flags}" linkflags="%{boost-link-flags}" install --prefix=\$(DESTDIR) \$(LIB_TARGETS)
    EOF
  build-commands:
  - make
  install-commands:
  - make -j 1 install DESTDIR="%{install-root}%{prefix}"

depends:
- sdk.bst
