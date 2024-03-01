# About

![](https://gitlab.com/squey/squey/-/raw/main/squey_screenshot.png)

## What is Squey?

[Squey](https://squey.org) (formerely INENDI Inspector) allows any user to perform interactive explorations over very large amounts of data.

Thanks to its innovative set of visualisations, the user is freed from the classical burden of query-based interactions and can then use all his logical and deductive power to discover valuable insights from the data.

It can be used for different purposes:
- Initial understanding of large and complex datasets
- Isolating weak signals very efficiently
- Controlling and improving Machine Learning algorithms

The ability to correlate several datasets with different structures permits to intuitively understand causal chains.

## Installation

### Linux

Installing the software (as a user):

<a href='https://flathub.org/apps/details/org.squey.Squey'><img width='190px' alt='Download on Flathub' src='https://flathub.org/assets/badges/flathub-badge-en.png'/></a>


```
flatpak install --user -y https://dl.flathub.org/repo/appstream/org.squey.Squey.flatpakref
```

Running the software :
* from the desktop environment : Simply click on the "Squey" shortcut
* from CLI locally : ```flatpak run org.squey.Squey```
* from CLI on a remote machine with SSH export display : ```flatpak run --command=bash org.squey.Squey -c "DISPLAY=$DISPLAY squey_launcher.sh"```

### Windows

Windows support is available through the use of WSL2:

https://squey.gitlab.io/squey/squey_installer.exe

### Container

Deploying the software as a service in a private cloud:

[![dockeri.co](https://dockerico.blankenship.io/image/squey/squey)](https://hub.docker.com/r/squey/squey)


Or building your own container image:

https://squey.gitlab.io/squey/squey_docker.zip


See ```README.md``` contained in archive file

## Reference manual

https://doc.squey.org

## Development

See the developement [README.md](buildstream/README.md) page.

### Installing a flatpak development branch

Note : Merge Requests having the `action::flatpak_export` [label](https://gitlab.com/squey/squey/-/labels#) are exported by the CI/CD pipeline as a flatpak package named after the git branch name.

Adding the flatpak development remote (once):
```
flatpak --user remote-add --no-gpg-verify squey_dev http://inspector-cassiopee.ensam.eu/flatpak
```
Installing a development branch:
```
flatpak install --user squey_dev org.squey.Squey//<dev_branch_name>
```

### Roadmap

https://gitlab.com/groups/squey/-/roadmap