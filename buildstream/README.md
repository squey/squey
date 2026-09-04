
# Development

## Clone the project

Note : you should have `git` and `git-lfs` installed.

```
git clone --recursive https://gitlab.com/squey/squey.git
```

## Build in the devcontainer

The quickest way in: a prebuilt image carrying the whole build sysroot, so the
host needs nothing but a container engine. Budget around 12 GB of disk for the
image. It produces a native Linux build only -- cross-compilation and packaging
go through the [development shell](#development-shell).

**With podman, set this once**, in `~/.config/containers/containers.conf`.
Without it the container user owns nothing in the workspace and cannot even
write to it. Docker needs nothing.

```
[containers]
userns = "keep-id"
```

Then open the repository in an editor supporting the
[Development Containers specification](https://containers.dev) -- VS Code,
GitHub Codespaces, JetBrains Gateway -- and accept "Reopen in Container".
Or use the [CLI](https://github.com/devcontainers/cli):

```
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . .devcontainer/configure_builds.sh
devcontainer exec --workspace-folder . cmake --build builds/x86_64-linux-gnu/Clang/RelWithDebInfo
```

An editor runs `configure_builds.sh` for you as the `postCreateCommand`; the CLI
does not, hence the explicit call. Running it twice costs nothing.

### Running the GUI

The Wayland socket is bound into the container, so the window lands on your
desktop like any other application:

```
builds/x86_64-linux-gnu/Clang/RelWithDebInfo/squey.sh
```

Pass `debug` to `squey.sh` to start it under gdb.

Rendering falls back to the CPU. For an NVIDIA GPU, open `.devcontainer/gpu/`
rather than the default configuration: an editor offers the choice when
reopening, and the CLI takes `--config .devcontainer/gpu/devcontainer.json`. It
is a separate configuration because it names devices that have to exist, so on a
machine without an NVIDIA card it would refuse to start.

### If the image will not pull

A branch pins the image of its own dependency graph, and that image is published
by the merge request pipeline. On a branch that never had one the pull fails
with `manifest unknown`: open the merge request and let the CI publish it, or
build it yourself.

```
buildstream/scripts/build_devcontainer_image.sh --push=true --update-pin=true
```

The tag is a digest of the dependency graph, so only a commit that changes a
dependency calls for a new image. The `ensure devcontainer image` CI job builds
and pushes that image by itself, on every merge request. The one thing it will
not do is write the new tag into your branch, so when a dependency moves it
stops and prints the one-line command that does -- the image is already waiting
by then. Staging the sysroot takes around 11 GB, so point `TMPDIR` at a disk
with room if `/tmp` is a tmpfs.

## Development shell

The BuildStream sandbox: slower to enter than the devcontainer, and the only way
to cross-compile and to produce a package.

Note : you should have `python` installed.

```
cd squey/buildstream && ./dev_shell.sh
```

From then you can choose to use the development shell to compile and run the software as such:

```
cd builds/{x86_64-linux-gnu,x86_64-apple-darwin,aarch64-apple-darwin}/{Clang,GCC}/{Debug,RelWithDebInfo} && cmake --build . [ && ./squey.sh ]
```

or you can continue to configure an IDE.

### Configure Visual Studio Code

Install the following extensions :

1. [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
2. [CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake)
3. [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
4. [CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb)

`dev_shell.sh` embeds an SSH server and a
[waypipe](https://gitlab.freedesktop.org/mstoeckl/waypipe) tunnel, so keep it
running: that is what lets vscode open the project and debug the GUI inside the
sandbox under Wayland.

Append the [preconfigured SSH host configuration](sshd/ssh_config.squey) to your
own, then connect with the `Remote Explorer` extension using an SSH key --
password login is not supported.

```
cat buildstream/sshd/ssh_config.squey | tee -a ~/.ssh/config
```

Each target has its own port, so several sandboxes can run at once:
`SqueyLinux` (6666), `SqueyWin` for `--target_triple=x86_64-w64-mingw32` (6667),
`SqueyMac` for `*-apple-darwin` (6668).

Remote Explorer            | Open folder
:-------------------------:|:-------------------------:
![](doc/vscode_remote_development_extension.png)  | ![](doc/vscode_open_folder.png)


CodeLLDB           | 
:-------------------------:
![](doc/vscode_codelldb_extension.png)

# Flatpak

## Generating and installing a local flatpak package

```
cd buildstream && ./build.sh --export=true --export-dir=local_repo
```

Adding the flatpak local remote (once):
```
flatpak --user remote-add --no-gpg-verify squey_local ./local_repo
```

Installing a local package:
```
flatpak install --user squey_local org.squey.Squey//main
```
