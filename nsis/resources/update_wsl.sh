#!/bin/bash

apt-get update
dpkg --configure -a
apt-get install -y flatpak