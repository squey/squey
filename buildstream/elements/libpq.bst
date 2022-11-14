kind: autotools
description: |

    PostgreSQL is a powerful, open source object-relational database system with over 30 years of active development that has earned it a strong reputation for reliability, feature robustness, and performance. 

sources:
- kind: tar
  url: https://ftp.postgresql.org/pub/source/v14.2/postgresql-14.2.tar.bz2
  ref: 2cf78b2e468912f8101d695db5340cf313c2e9f68a612fb71427524e8c9a977a

config:
    configure-commands:
        - ./configure %{conf-args} --with-ssl=openssl
    build-commands:
        - cd src/interfaces/libpq
        - make
    install-commands:
        - cd src/interfaces/libpq
        - make -j1 DESTDIR="%{install-root}" install

depends:
- base.bst
- freedesktop-sdk.bst:components/openssl.bst
