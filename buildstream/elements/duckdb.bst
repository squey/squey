kind: cmake
description: |

    DuckDB is an in-process SQL OLAP database management system. It is used to expose the pvcop columns as a virtual table so that selections can be generated from SQL queries.

sources:
- kind: tar
  url: github:duckdb/duckdb/archive/refs/tags/v1.5.5.tar.gz
  ref: f33155ff962e6e1e08fd1e9caffa487d4325aa60999e2eabc76feff534d6558b

(@): buildstream/elements/base/target.yml

(?):
  # DuckDB detects its platform string by compiling a small binary *for the
  # target* and then running it, which cannot work when cross-compiling: the
  # Mach-O or PE binary fails with "Exec format error" on the Linux host. Their
  # own build tells you the way out -- "Provide explicit DUCKDB_PLATFORM to
  # avoid build-type detection of the platform" -- so name it per target.
  # Note the CMake variable is DUCKDB_EXPLICIT_PLATFORM: the DUCKDB_PLATFORM
  # the message names is only read from the environment, and passing it with
  # -D has no effect.
  - target_triple == "x86_64-linux-gnu":
      variables:
        duckdb-platform: "linux_amd64"
  - target_triple == "x86_64-apple-darwin":
      variables:
        duckdb-platform: "osx_amd64"
  - target_triple == "aarch64-apple-darwin":
      variables:
        duckdb-platform: "osx_arm64"
  - target_triple == "x86_64-w64-mingw32":
      variables:
        duckdb-platform: "windows_amd64"
      # The DLL is unusable from C++ under MinGW: winapi.hpp expands DUCKDB_API
      # to nothing there -- its guard reads `defined(_WIN32) &&
      # !defined(__MINGW32__)` -- so not one C++ declaration carries
      # __declspec(dllexport), while duckdb.h keeps exporting the C API. That
      # single explicit export is enough to switch off the auto-export ld falls
      # back to when a PE image marks no symbol at all, so the import library
      # resolves the C API and nothing else. Squey links duckdb_static instead,
      # the same target the upstream shell links; drop the pair so that
      # find_library() cannot pick it and the Windows package does not carry a
      # dead copy of the whole engine.
      config:
        install-commands:
          (>):
          - |
            rm -f "%{install-root}%{bindir}/libduckdb.dll" "%{install-root}%{libdir}/libduckdb.dll.a"
  # The ICU extension is built against the ICU Squey already ships, rather than
  # the copy DuckDB carries (see WITH_INTERNAL_ICU below): the SDK's on Linux,
  # the one icu.bst cross-builds elsewhere.
  - target_triple == "x86_64-linux-gnu":
      build-depends:
        (>):
        - freedesktop-sdk.bst:components/icu.bst
  - target_triple != "x86_64-linux-gnu":
      build-depends:
        (>):
        - icu.bst

variables:
  source-dir: "."
  # ENABLE_SANITIZER and ENABLE_UBSAN both default to TRUE upstream, and
  # BUILD_SHELL/BUILD_UNITTESTS pull in the CLI and the test runner, which more
  # than doubles the build time. OVERRIDE_GIT_DESCRIBE is required because the
  # release tarball carries no .git directory for the version to be derived from.
  # CMAKE_CXX_STANDARD is theirs to default to 11, where a static constexpr
  # member is not implicitly inline: the library then emits a strong definition
  # for each LogicalType constant, while a C++17 translation unit naming one
  # emits its own into a COMDAT section, and ld merges neither into the other.
  # Only PE calls that a multiple definition -- ELF quietly lets the strong one
  # win -- so build the library in the standard its consumer compiles against.
  # BUILD_EXTENSIONS=icu links the ICU extension in, beside the core functions
  # and parquet the build always links. to_timestamp() answers a TIMESTAMP WITH
  # TIME ZONE, and turning one into a DATE, a string (strftime) or a truncated
  # instant (date_trunc) is ICU's work: without it DuckDB tries to autoload the
  # extension, which Squey turns off, and the query fails. Linked in, it loads
  # with the database -- on Windows through the lib*_extension.a Squey collects.
  # WITH_INTERNAL_ICU=OFF builds it against the ICU the package already holds
  # for pvcop and Qt, instead of the copy DuckDB vendors: ICU 66, from 2020,
  # with its own collation and time zone data, and 10 MB more in libduckdb.
  # One ICU in the process, and the one that gets updated.
  cmake-local: >-
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHELL=OFF
    -DBUILD_UNITTESTS=OFF
    -DBUILD_BENCHMARKS=OFF
    -DENABLE_SANITIZER=OFF
    -DENABLE_UBSAN=OFF
    -DENABLE_EXTENSION_AUTOLOADING=OFF
    -DENABLE_EXTENSION_AUTOINSTALL=OFF
    -DBUILD_EXTENSIONS=icu
    -DWITH_INTERNAL_ICU=OFF
    -DOVERRIDE_GIT_DESCRIBE=v1.5.5
    -DDUCKDB_EXPLICIT_PLATFORM=%{duckdb-platform}
    -DCMAKE_CXX_STANDARD=17

config:
  configure-commands:
    # Upstream unconditionally turns CMAKE_EXPORT_COMPILE_COMMANDS on and adds a
    # 'clangd_cache' target that writes to "${CMAKE_BINARY_DIR}/../../.cache",
    # assuming the build directory sits two levels below the source root (their
    # own build/debug layout). BuildStream configures in-tree, so that path
    # escapes the staged sources and lands on a read-only directory, failing the
    # build once every object has been compiled. Turning the flag off in the
    # source drops the target: the `set()` is a plain assignment, so passing
    # -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF would be overridden.
    - |
      sed -i 's/set(CMAKE_EXPORT_COMPILE_COMMANDS ON)/set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)/' CMakeLists.txt
      %{cmake}

build-depends:
- sdk.bst
