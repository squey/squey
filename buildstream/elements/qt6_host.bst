kind: cmake

variables:
    qt-crossplatform-options: ''
    cmake-local-extra: ''

(@): buildstream/elements/qt6.yml

build-depends:
  (>):
  - base.bst
  - freedesktop-sdk.bst:components/pkg-config.bst

depends:
  (>):
  - freedesktop-sdk.bst:components/openssl.bst
  - mariadb-connector-c_host.bst
  - libpq_host.bst