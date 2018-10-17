#!/bin/bash

apt-get update
dpkg --configure -a
apt-get dist-upgrade -y
do-release-upgrade -f DistUpgradeViewNonInteractive