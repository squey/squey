[optional] Step 1 - install NVIDIA drivers
==========================================

In order to have GPU access inside the container, you will need to install nvidia drivers on the host machine.

See https://devblogs.nvidia.com/gpu-containers-runtime/ for more information on GPU containers

Step 2 - Ensure that user namespaces are enabled
================================================

Flatpak needs to have this Linux kernel feature enabled on the host machine.
On most systems, it may be enabled as following :

$ echo 'kernel.unprivileged_userns_clone=1' | sudo tee -a /etc/sysctl.d/00-local-userns.conf
$ echo 'user.max_user_namespaces=1000' | sudo tee -a /etc/sysctl.d/00-local-userns.conf
$ sudo sysctl --system

[optional] Step 3 - download_files.sh
=====================================

If the host is located in an air-gap, this script will download all the needed resources.
This script is to be run on an internet connected machine.

If your host machine is directly connected to the Internet, there is no need for this extra step.

Step 4 - build.sh
=================

Customize the "env.conf" file containing various configuration variables and
start the build of the container image.
Once properly built, the container image name is "squey/squey".

Step 5 - run.sh
===============

Run the container.
You can access the application in a web browser at the following location: https://<container_hostname>:8443
Once running, the container name is "squey".

Step 6 - update.sh
==================

Update the application.
If the host is located in an air-gap, you will need to execute "download_files.sh" on an internet connected machine before.
