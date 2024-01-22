
# Development

## Clone the project

Note : you should have `git` and `git-lfs` installed.

```
git clone --recursive https://gitlab.com/squey/squey.git
```

## Development shell

Note : you should have `python` installed.

```
cd squey/buildstream && ./dev_shell.sh
```

From then you can choose to use the development shell to compile and run the software as such:

```
cd builds/{Clang,GCC}/{Debug,RelWithDebInfo} && cmake --build . && ./squey.sh
```

or you can continue to configure an IDE.

## Configure Visual Studio Code

Install the following extensions :

1. [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
2. [CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake)
3. [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
4. [CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb)

The `dev_shell.sh` script is embedding a local SSH server as well as a [waypipe](https://gitlab.freedesktop.org/mstoeckl/waypipe) tunnel that permit to debug the GUI application inside BuildStream development sandbox under Wayland. So don't forget to have it running to be able to open the project in vscode :

```
cd squey/buildstream && ./dev_shell.sh
```

Just append the [preconfigured SSH host configuration](sshd/ssh_config.squey) to your local SSH configuration file:

```
cat buildstream/sshd/ssh_config.squey | tee -a ~/.ssh/config
```

Then use the vscode `Remote Explorer` extension to connect to the `Squey` hostname using a ssh key (note: password login is not supported).

Remote Explorer            | Open folder
:-------------------------:|:-------------------------:
![](doc/vscode_remote_development_extension.png)  | ![](doc/vscode_open_folder.png)


CodeLLDB           | 
:-------------------------:
![](doc/vscode_codelldb_extension.png)

# Flatpak

## Generating and installing a local flatpak package

```
cd buildstream && ./build.sh --flatpak-export=true --flatpak-repo=local_repo
```

Adding the flatpak local remote (once):
```
flatpak --user remote-add --no-gpg-verify squey_local ./local_repo
```

Installing a local package:
```
flatpak install --user squey_local org.squey.Squey//main
```