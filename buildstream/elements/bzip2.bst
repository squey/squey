kind: manual

sources:
- kind: git_repo
  url: sourceware:git/bzip2.git
  track: bzip2-*
  ref: bzip2-1.0.8-0-g6a8690fc8d26c815e798c588f796eabe9d684cf0

(@): buildstream/elements/base/target.yml

config:
  (?):
  - (target_triple != "x86_64-linux-gnu"):
        build-commands:
        - |
          sed 's/libbz2.so.\([[:digit:]]\.[[:digit:]]\.[[:digit:]]\)/libbz2.\1.%{dylib-extension}/g' -i Makefile-libbz2_so
          sed 's/libbz2.so.\([[:digit:]]\.[[:digit:]]\)/libbz2.\1.%{dylib-extension}/g' -i Makefile-libbz2_so
          if [ "%{target-platform}" = "darwin" ] ; then
            sed 's/-soname/-install_name/g' -i Makefile-libbz2_so
          elif [ "%{target-platform}" = "windows" ] ; then
            sed 's/-shared /-shared -Wl,--out-implib,libbz2.%{dylib-extension}.a /g' -i Makefile-libbz2_so
            cat Makefile-libbz2_so
          fi
          make -f Makefile-libbz2_so CC="%{cc-compiler} $CFLAGS $LDFLAGS"
          make bzip2recover CC="%{cc-compiler} $CFLAGS $LDFLAGS" AR="%{toolchain-dir}/%{host}-ar" RANLIB="%{toolchain-dir}/%{host}-ranlib"
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
        install-commands:
          (>):
          - |
            %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libbz2.1.0.%{dylib-extension}" "%{install-root}%{libdir}/libbz2.%{dylib-extension}"

  install-commands:
  - |
    LIB_DIR=%{libdir}
    if [ "%{target-platform}" = "windows" ] ; then
      LIB_DIR=%{bindir}
      install -D -m 755 libbz2.%{dylib-extension}.a --target-directory %{install-root}%{libdir}
      install -D -m 755 libbz2.1.0.%{dylib-extension} %{install-root}%{bindir}/libbz2.1.0.8.%{dylib-extension}
    else
      full_name="$(readlink libbz2.1.0.%{dylib-extension})"
      install -D -m 755 "${full_name}" --target-directory %{install-root}${LIB_DIR}
      ln -s "${full_name}" %{install-root}${LIB_DIR}/libbz2.1.0.%{dylib-extension}
      ln -s "${full_name}" %{install-root}${LIB_DIR}/libbz2.%{dylib-extension}
    fi
    install -D -m 755 bzip2-shared%{exe-extension} %{install-root}%{bindir}/bzip2%{exe-extension}
    ln -s bzip2%{exe-extension} %{install-root}%{bindir}/bzcat%{exe-extension}
    ln -s bzip2%{exe-extension} %{install-root}%{bindir}/bunzip2%{exe-extension}
    install -D -m 755 bzip2recover%{exe-extension} --target-directory %{install-root}%{bindir}
    install -D -m 755 bzmore --target-directory %{install-root}%{bindir}
    ln -s bzmore %{install-root}%{bindir}/bzfless
    install -D -m 755 bzgrep --target-directory %{install-root}%{bindir}
    ln -s bzgrep %{install-root}%{bindir}/bzegrep
    ln -s bzgrep %{install-root}%{bindir}/bzfgrep
    install -D -m 755 bzdiff --target-directory %{install-root}%{bindir}
    ln -s bzdiff %{install-root}%{bindir}/bzcmp
    install -D -m 755 bzlib.h --target-directory %{install-root}%{includedir}
    install -D -m 755 bzgrep.1 bzmore.1 bzdiff.1 --target-directory %{install-root}%{mandir}/man1

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{libdir}/libbz2.%{dylib-extension}'

build-depends:
- sdk.bst
