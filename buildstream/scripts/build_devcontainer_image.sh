#!/bin/bash
#
# Builds, and optionally publishes, the devcontainer image: the staged build
# dependencies of squey.bst turned into a container root filesystem, so that a
# contributor can compile Squey without ever running BuildStream.
#
# The image is the degraded but trivial path. It only covers a native Linux
# build: cross-compilation to Windows and macOS, the flatpak/MSIX/DMG packaging
# and the release pipeline all stay in BuildStream, which remains the only way
# to produce something shippable.
#
# It carries no source and no build directory. The workspace is bind mounted by
# the devcontainer runtime, and .devcontainer/devcontainer.json pins the tag
# matching the dependency graph of the branch it lives on, the same way
# .gitlab-ci.yml pins BASE_IMAGE_NAME.

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

usage() {
  echo "Usage: $0"
  echo "--registry=<image_repository>  (default: \$DEVCONTAINER_REGISTRY, then the GitLab CI one)"
  echo "--push=<true/false>            publish the image once built"
  echo "--update-pin=<true/false>      rewrite the image reference in .devcontainer/devcontainer.json"
  echo "--force=<true/false>           rebuild even when the registry already holds the tag"
  echo "--storage-root=<directory>     where podman keeps its images (default: \$PODMAN_STORAGE_ROOT)" 1>&2
  exit 1
}

REGISTRY="${DEVCONTAINER_REGISTRY:-${CI_REGISTRY_IMAGE:+$CI_REGISTRY_IMAGE/devcontainer}}"
PUSH=false
UPDATE_PIN=false
FORCE=false
STORAGE_ROOT="${PODMAN_STORAGE_ROOT:-}"

OPTS=$(getopt -o h --long help,registry:,push:,update-pin:,force:,storage-root: -n 'parse-options' -- "$@")
if [ $? != 0 ] ; then usage >&2 ; fi
eval set -- "$OPTS"
while true; do
  case "$1" in
    -h | --help ) usage >&2 ;;
    --registry ) REGISTRY="$2"; shift 2 ;;
    --push ) PUSH="$2"; shift 2 ;;
    --update-pin ) UPDATE_PIN="$2"; shift 2 ;;
    --force ) FORCE="$2"; shift 2 ;;
    --storage-root ) STORAGE_ROOT="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ -z "$REGISTRY" ]; then
  echo >&2 "No image repository: pass --registry or set DEVCONTAINER_REGISTRY."
  exit 1
fi

command -v podman &> /dev/null || { echo >&2 "'podman' executable not found"; exit 1; }

# Every filesystem a container sees is an overlayfs, and the overlay driver of
# podman refuses to stack on one without a fuse mount program. Running the build
# inside a container therefore needs its image store somewhere else: a directory
# on a volume the host really backs, which CI has in $CI_BUILDS_DIR. Should that
# one be an overlayfs too, fall back to vfs, which works anywhere at the cost of
# unpacking every layer -- a mild price here, as the image is a single layer.
PODMAN_OPTS=()
if [ -n "$STORAGE_ROOT" ]; then
  mkdir -p "$STORAGE_ROOT"
  PODMAN_OPTS+=(--root "$STORAGE_ROOT")
fi
if ! podman "${PODMAN_OPTS[@]}" info &> /dev/null; then
  echo "podman cannot use ${STORAGE_ROOT:-its default store}, falling back to the vfs driver."
  PODMAN_OPTS+=(--storage-driver=vfs)
fi

TAG="$(scripts/devcontainer_image_tag.sh)"
IMAGE="$REGISTRY:$TAG"
echo "Devcontainer image: $IMAGE"

# Publishing is idempotent: the tag is a pure function of the dependency graph,
# so an image already in the registry is byte-for-byte the one this run would
# rebuild. Letting the job run on every pipeline and exit here is more robust
# than gating it on "rules:changes", which misses a rebased or squashed branch.
# skopeo only saves a rebuild here, so its absence costs time rather than
# correctness: pushing an image the registry already holds is a no-op.
if [ "$FORCE" != true ] && [ "$PUSH" = true ] && command -v skopeo &> /dev/null \
   && skopeo inspect "docker://$IMAGE" &> /dev/null; then
  echo "Already published, nothing to do."
  if [ "$UPDATE_PIN" = true ]; then
    "$DIR/scripts/update_devcontainer_pin.sh" "$IMAGE"
  fi
  exit 0
fi

SYSROOT_DIR="$(mktemp -d)"
function cleanup {
  rm -rf -- "$SYSROOT_DIR"
}
trap cleanup EXIT

# .common.sh validates TARGET_TRIPLE against a hardcoded list and reads $? right
# after the check, which a caller running under "set -e" never gets to see: the
# failing grep kills the script first. Set the variable it expects and lift
# errexit for the duration of the source, as build.sh does by sourcing it before
# turning errexit on.
export TARGET_TRIPLE=x86_64-linux-gnu
set +e
{ source .common.sh ; } 1>&2
set -e

# Deliberately not "--hardlinks": the checkout would then share its inodes with
# the local CAS, and the overlay below writes into the tree. A stray write
# through a hardlink corrupts a cached artifact for every later build.
# "--no-integrate" leaves out the integration commands, which expect a running
# sandbox and have nothing to do in a container image.
bst --option target_triple x86_64-linux-gnu --option cxx_compiler clang++ \
    artifact checkout --deps build --no-integrate \
    --directory "$SYSROOT_DIR" squey.bst

# BuildStream mounts /etc/passwd and /etc/group from the host into its sandbox
# (see the "host-files" of app.yml), so the staged sysroot holds neither. A
# container needs its own: create the unprivileged user the devcontainer runs
# as. VS Code remaps its uid to the host one through "updateRemoteUserUID",
# which is what keeps the files it writes in the workspace owned by the user.
# The image owns the two paths devcontainer.json puts a volume on, rather than
# leaving the runtime to invent them. A build driven through the devcontainer
# CLI once died on "ccache: error: Permission denied" against a volume created
# for it; that has not been pinned down to a cause -- mounting a volume over a
# missing path reproduces nothing on its own -- but a runtime seeding a volume
# from a directory that exists, with an owner, has one less thing to guess.
install -d "$SYSROOT_DIR/etc" "$SYSROOT_DIR/home/dev" \
          "$SYSROOT_DIR/home/dev/.cache/ccache" "$SYSROOT_DIR/home/dev/.squey"
cat > "$SYSROOT_DIR/etc/passwd" << 'EOF'
root:x:0:0:root:/root:/usr/bin/bash
dev:x:1000:1000:Squey developer:/home/dev:/usr/bin/bash
nobody:x:65534:65534:nobody:/:/usr/sbin/nologin
EOF
cat > "$SYSROOT_DIR/etc/group" << 'EOF'
root:x:0:
dev:x:1000:
nogroup:x:65534:
EOF
chown -R 1000:1000 "$SYSROOT_DIR/home/dev" 2> /dev/null || true

# run_cmd.sh, the wrapper both the packaged application and the test suite go
# through, writes the OpenCL vendor files to /etc/opencl_vendors and creates
# that directory itself -- which an unprivileged user cannot do under /etc.
# Ship it, owned by the container user, so the wrapper runs unmodified and the
# GPU is reachable from the container (see buildstream/README.md).
install -d "$SYSROOT_DIR/etc/opencl_vendors"
chown 1000:1000 "$SYSROOT_DIR/etc/opencl_vendors" 2> /dev/null || true

# The nraw_tmp of pvconfig.ini, where Squey spills the imported data. It greets
# whoever it does not find it writable with a "chose a temporary files
# directory" dialog, on every single start, so ship it writable rather than
# leave every container asking. The development shell has the same directory
# mounted in by .common.sh.
install -d "$SYSROOT_DIR/srv/tmp-squey"
chown 1000:1000 "$SYSROOT_DIR/srv/tmp-squey" 2> /dev/null || true

# Every devcontainer runtime reads /etc/os-release to work out what it is
# talking to, and the staged sysroot carries none: the probe fails, and the
# tooling falls back to guessing. Name it for what it is rather than borrow a
# distribution identity the image does not have.
cat > "$SYSROOT_DIR/etc/os-release" << 'EOF'
NAME="Squey build sysroot"
ID=squey-devcontainer
ID_LIKE=freedesktop-sdk
PRETTY_NAME="Squey build sysroot (freedesktop-sdk)"
HOME_URL="https://squey.org"
EOF

# The python version is read off the sysroot rather than hardcoded, as PYTHONPATH
# has to point at the site-packages the staged interpreter actually looks for.
PYTHON_SITE_PACKAGES="$(cd "$SYSROOT_DIR" && echo app/lib/python*/site-packages)"

# Mirrors the environment BuildStream gives a "bst shell --build" (the PATH and
# PKG_CONFIG_PATH of app.yml, and /etc/target_env_vars.sh which the sysroot
# already carries), so that a shell in the container sees what the dev shell
# sees. What depends on the workspace path is left to devcontainer.json.
tar --numeric-owner -C "$SYSROOT_DIR" -c . | podman "${PODMAN_OPTS[@]}" import \
  --change 'ENV PATH=/app/bin:/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin' \
  --change 'ENV PKG_CONFIG_PATH=/app/lib/pkgconfig:' \
  --change 'ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/app/lib' \
  --change "ENV PYTHONPATH=/$PYTHON_SITE_PACKAGES" \
  --change 'ENV PREFIX=/app' \
  --change 'ENV TARGET_TRIPLE=x86_64-linux-gnu' \
  --change 'ENV TARGET_PLATFORM=linux' \
  --change 'ENV HOST=x86_64-unknown-linux-gnu' \
  --change 'ENV TOOLCHAIN_DIR=/usr/bin' \
  --change 'ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
  --change 'ENV CCACHE_DIR=/home/dev/.cache/ccache' \
  --change 'CMD ["/usr/bin/bash"]' \
  - "$IMAGE"

echo "Built $IMAGE"

if [ "$PUSH" = true ]; then
  podman "${PODMAN_OPTS[@]}" push "$IMAGE"
  echo "Pushed $IMAGE"
fi

if [ "$UPDATE_PIN" = true ]; then
  "$DIR/scripts/update_devcontainer_pin.sh" "$IMAGE"
fi
