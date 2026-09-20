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
  echo "--force=<true/false>           rebuild even when the registry already holds the tag" 1>&2
  exit 1
}

REGISTRY="${DEVCONTAINER_REGISTRY:-${CI_REGISTRY_IMAGE:+$CI_REGISTRY_IMAGE/devcontainer}}"
PUSH=false
UPDATE_PIN=false
FORCE=false

OPTS=$(getopt -o h --long help,registry:,push:,update-pin:,force: -n 'parse-options' -- "$@")
if [ $? != 0 ] ; then usage >&2 ; fi
eval set -- "$OPTS"
while true; do
  case "$1" in
    -h | --help ) usage >&2 ;;
    --registry ) REGISTRY="$2"; shift 2 ;;
    --push ) PUSH="$2"; shift 2 ;;
    --update-pin ) UPDATE_PIN="$2"; shift 2 ;;
    --force ) FORCE="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ -z "$REGISTRY" ]; then
  echo >&2 "No image repository: pass --registry or set DEVCONTAINER_REGISTRY."
  exit 1
fi

command -v skopeo &> /dev/null || { echo >&2 "'skopeo' executable not found"; exit 1; }
command -v jq &> /dev/null || { echo >&2 "'jq' executable not found"; exit 1; }

# The image is assembled as an OCI layout and handed to skopeo, rather than
# imported into a container store and pushed out of it. A store keeps layers
# unpacked, so anything leaving one has to be compressed again on the way out:
# the sysroot was compressed twice, unpacked once in between, and three copies
# of it existed at the peak -- on a runner with room for two.
#
# Writing the layout by hand costs the lines below and removes all of that. It
# also settles which storage driver to use, by needing none: /builds sits on a
# nodev filesystem, where the overlay driver cannot create the device nodes it
# marks deletions with.
COMPRESSOR=(gzip -1)
command -v pigz &> /dev/null && COMPRESSOR=(pigz -1)
echo "Compressing with ${COMPRESSOR[0]}."

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
OCI_DIR="$SYSROOT_DIR.oci"
function cleanup {
  # The checkout goes as soon as the layer exists, and the layer is moved into
  # the layout rather than copied, so these two are never both full at once.
  # Whatever survives a failure is removed here: on a runner this directory is
  # shared with the clone and with the other slots.
  rm -rf -- "$OCI_DIR" \
    || echo >&2 "warning: $OCI_DIR survived and still holds the image layer."
  rm -rf -- "$SYSROOT_DIR" \
    || echo >&2 "warning: $SYSROOT_DIR survived and still holds the staged sysroot."
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

# The artifacts the image is made of only exist once built, which is the doing
# of the Linux build -- and that job and this one take turns on the same cache,
# in no set order. Whenever a dependency moved and this job came first, it
# failed on artifacts nobody had built yet; on the default branch, where it runs
# alone, nothing else would ever build them. So build whatever is missing: what
# the Linux build already has is a cache hit.
BUILD_DEPS="$(bst --option target_triple x86_64-linux-gnu --option cxx_compiler clang++ \
    show --deps build --format '%{name}' squey.bst)"
# shellcheck disable=SC2086 # one element name per word
bst --option target_triple x86_64-linux-gnu --option cxx_compiler clang++ build $BUILD_DEPS

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

# A shell in the container should see what a "bst shell --build" sees, and every
# one of those values is already written down somewhere: the sysroot carries
# /etc/target_env_vars.sh, BuildStream composes PATH and PKG_CONFIG_PATH itself,
# and env.conf holds what .common.sh exported on the way in. Retyping them here
# is what once left the container with its library path the wrong way round --
# two libLLVM live in this sysroot, only one of them is the one PortableCL was
# linked against, and the first one found wins. So read them.
#
# What is genuinely this image's own decision stays below: where ccache writes,
# and a library path that prefers the sysroot over the SDK and names the
# multiarch directory the SDK actually uses.
source "$SYSROOT_DIR/etc/target_env_vars.sh"
BST_ENV="$(bst --option target_triple x86_64-linux-gnu --option cxx_compiler clang++ \
           show --deps none --format '%{env}' squey.bst 2> /dev/null)"
function bst_env {
  sed -n "s/^$1: *//p" <<< "$BST_ENV" | tr -d "'"
}
for required in PREFIX TARGET_TRIPLE TARGET_PLATFORM HOST TOOLCHAIN_DIR; do
  [ -n "${!required}" ] || { echo >&2 "$required is not set by the sysroot"; exit 1; }
done
for required in PATH PKG_CONFIG_PATH; do
  [ -n "$(bst_env "$required")" ] || { echo >&2 "bst declares no $required"; exit 1; }
done

# One pass over the sysroot yields both digests the OCI format asks for: the
# layer descriptor identifies the compressed blob, the config identifies the
# uncompressed stream it unpacks to. The fifo is what keeps it to one pass -- a
# process substitution would leave sha256sum still running when the pipeline
# returns, and the digest half written.
mkdir -p "$OCI_DIR/blobs/sha256"
DIFF_FIFO="$OCI_DIR/diff.fifo"
mkfifo "$DIFF_FIFO"
sha256sum < "$DIFF_FIFO" | cut -d' ' -f1 > "$OCI_DIR/diff_id" &
DIFF_PID=$!
tar --numeric-owner -C "$SYSROOT_DIR" -c . \
  | tee "$DIFF_FIFO" \
  | "${COMPRESSOR[@]}" > "$OCI_DIR/layer"
wait "$DIFF_PID"
rm -f -- "$DIFF_FIFO"

# A dozen gigabytes, and nothing reads them after this.
rm -rf -- "$SYSROOT_DIR"

DIFF_ID="$(cat "$OCI_DIR/diff_id")"
rm -f -- "$OCI_DIR/diff_id"
LAYER_DIGEST="$(sha256sum "$OCI_DIR/layer" | cut -d' ' -f1)"
LAYER_SIZE="$(stat -c %s "$OCI_DIR/layer")"
mv "$OCI_DIR/layer" "$OCI_DIR/blobs/sha256/$LAYER_DIGEST"

jq -n --arg diff "sha256:$DIFF_ID" --arg pysite "/$PYTHON_SITE_PACKAGES" \
      --arg path "$(bst_env PATH)" --arg pkgconfig "$(bst_env PKG_CONFIG_PATH)" \
      --arg ldpath "/usr/lib/$TARGET_TRIPLE:$PREFIX/lib" \
      --arg prefix "$PREFIX" --arg triple "$TARGET_TRIPLE" \
      --arg platform "$TARGET_PLATFORM" --arg host "$HOST" \
      --arg toolchain "$TOOLCHAIN_DIR" \
      --arg protobuf "$PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" '{
  created: (now | todate),
  architecture: "amd64",
  os: "linux",
  config: {
    Env: [
      "PATH=" + $path,
      "PKG_CONFIG_PATH=" + $pkgconfig,
      "LD_LIBRARY_PATH=" + $ldpath,
      "PYTHONPATH=" + $pysite,
      "PREFIX=" + $prefix,
      "TARGET_TRIPLE=" + $triple,
      "TARGET_PLATFORM=" + $platform,
      "HOST=" + $host,
      "TOOLCHAIN_DIR=" + $toolchain,
      "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=" + $protobuf,
      "CCACHE_DIR=/home/dev/.cache/ccache"
    ],
    Cmd: ["/usr/bin/bash"]
  },
  rootfs: { type: "layers", diff_ids: [$diff] },
  history: [{ created: (now | todate), created_by: "bst artifact checkout --deps build squey.bst" }]
}' > "$OCI_DIR/config.json"
CONFIG_DIGEST="$(sha256sum "$OCI_DIR/config.json" | cut -d' ' -f1)"
CONFIG_SIZE="$(stat -c %s "$OCI_DIR/config.json")"
mv "$OCI_DIR/config.json" "$OCI_DIR/blobs/sha256/$CONFIG_DIGEST"

jq -n --arg cd "sha256:$CONFIG_DIGEST" --argjson cs "$CONFIG_SIZE" \
      --arg ld "sha256:$LAYER_DIGEST" --argjson ls "$LAYER_SIZE" '{
  schemaVersion: 2,
  mediaType: "application/vnd.oci.image.manifest.v1+json",
  config: { mediaType: "application/vnd.oci.image.config.v1+json", digest: $cd, size: $cs },
  layers: [{ mediaType: "application/vnd.oci.image.layer.v1.tar+gzip", digest: $ld, size: $ls }]
}' > "$OCI_DIR/manifest.json"
MANIFEST_DIGEST="$(sha256sum "$OCI_DIR/manifest.json" | cut -d' ' -f1)"
MANIFEST_SIZE="$(stat -c %s "$OCI_DIR/manifest.json")"
mv "$OCI_DIR/manifest.json" "$OCI_DIR/blobs/sha256/$MANIFEST_DIGEST"

echo '{"imageLayoutVersion": "1.0.0"}' > "$OCI_DIR/oci-layout"
jq -n --arg md "sha256:$MANIFEST_DIGEST" --argjson ms "$MANIFEST_SIZE" --arg tag "$TAG" '{
  schemaVersion: 2,
  mediaType: "application/vnd.oci.image.index.v1+json",
  manifests: [{
    mediaType: "application/vnd.oci.image.manifest.v1+json",
    digest: $md, size: $ms,
    annotations: { "org.opencontainers.image.ref.name": $tag }
  }]
}' > "$OCI_DIR/index.json"

echo "Built $IMAGE"

if [ "$PUSH" = true ]; then
  # The blob goes up exactly as it was written. skopeo copies it; it does not
  # unpack and recompress it the way a push out of a container store would.
  skopeo copy "oci:$OCI_DIR:$TAG" "docker://$IMAGE"
  echo "Pushed $IMAGE"
else
  # Nowhere to send it, so leave it where it can be looked at or loaded, and
  # say where rather than deleting it silently on the way out.
  KEEP_DIR="${TMPDIR:-/tmp}/devcontainer-oci-$TAG"
  rm -rf -- "$KEEP_DIR"
  mv "$OCI_DIR" "$KEEP_DIR"
  echo "Not pushing. The image is an OCI layout in $KEEP_DIR; load it with:"
  echo "  skopeo copy oci:$KEEP_DIR:$TAG containers-storage:$IMAGE"
fi

if [ "$UPDATE_PIN" = true ]; then
  "$DIR/scripts/update_devcontainer_pin.sh" "$IMAGE"
fi
