# About


## What is INENDI Inspector?

INENDI Inspector allows any user to perform interactive explorations over very large amounts of data.

Thanks to its innovative set of visualisations, the user is freed from the classical burden of query-based interactions and can then use all his logical and deductive power to discover valuable insights from the data.

The ability to correlate several datasets with different structures permits to intuitively understand the causal chains over the course of the whole life cycle of the studied product.

It can be used for different purposes:
- Initial understanding of large and complex datasets
- Isolating weak signals very efficiently
- Controlling and improving Machine Learning algorithms

## Installation

### Linux

Installing the software (as a user) :

```
flatpak install --user -y https://inendi.gitlab.io/inspector/install.flatpakref
```

Running the software :
* from the desktop environment : Simply click on the "INENDI Inspector" shortcut
* from CLI locally : ```flatpak run org.inendi.Inspector```
* from CLI on a remote machine with SSH export display : ```flatpak run --command=bash org.inendi.Inspector -c "DISPLAY=$DISPLAY inspector_launcher.sh"```

### Windows

Windows support is available through the use of WSL2 :

https://inendi.gitlab.io/inspector/inendi-inspector_installer.exe

### Docker

Deploying the software as a service in a private cloud:

```
https://inendi.gitlab.io/inspector/inendi-inspector_docker.zip
```

See ```README.md``` contained in archive file

## Development

### Getting a development shell

```
git clone --recursive https://gitlab.com/inendi/inspector.git
cd inspector/buildstream && ./dev_shell.sh
```

Compiling and running the software from the development shell:

```
cd {debug,release}_build && cmake --build . && ./inspector.sh
```

### Generating a flatpak package

```
cd buildstream && ./build.sh --repo=flatpak_repo --gpg-private-key-path=/path/to/private.pgp --gpg-sign-key=<sign_key>
```
