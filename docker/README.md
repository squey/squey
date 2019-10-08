Step 1 - install nvidia-docker2
===============================

In order to have GPU access inside the container, you will need to install the Docker "nvidia" runtime with nvidia-docker2.
The procedure is documented here : https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)

See https://devblogs.nvidia.com/gpu-containers-runtime/ for more information on GPU containers

Step 2 - download_files.sh [optional]
=====================================

If the host is located in an air-gap, this script will download all the needed resources.
This script is to be run on an internet connected machine.

Step 3 - build.sh
=================

Start the build of the Docker image.
Once properly built, the Docker image name is "inendi/inspector".
Before running this script, you can customize the "env.conf" file containing various configuration variables.

Step 4 - run.sh
===============

Run the Docker container.
You can access the application in a web browser at the following location: https://<container_hostname>
Once running, the Docker container name is "inendi-inspector".

Step 5 - update.sh
==================

Update the application.
If the host is located in an air-gap, you will need to execute "download_files.sh" on an internet connected machine before.
