#!/bin/bash

if [ $# -ne 1 ]; then
	echo "usage : `basename $0` <flatpakref URL>"
    exit 1
fi

apt-get -y remove --purge snap* git* vim* lxd* *openssh* man* krb5* lsof htop ed nano mlocate rsync parted ntfs-3g xfsprogs btrfs-progs dosfstools pciutils cloud-init tcpdump geoip-database gawk lvm2 iproute2 rsyslog strace keyboard-configuration shared-mime-info command-not-found-data busybox* groff-base language-selector-common iptables zerofree ubuntu-advantage-tools tmux telnet screen
apt-get -y autoremove
apt-get update
apt-get -y dist-upgrade
apt-get -y install sed wget libappstream-glib8 libjson-glib-1.0-0 libgpgme11 xdg-desktop-portal
apt-get clean
wget https://repo.esi-inendi.com/wsl/ostree_2099.4-1_amd64.deb
wget https://repo.esi-inendi.com/wsl/flatpak_1.99-1_amd64.deb
dpkg -i ostree*.deb && rm -f ostree*.deb
dpkg -i flatpak*.deb && rm -f flatpak*.deb
apt-get -f install
apt-get remove -y libfreetype6
useradd -m inendi
su - inendi -c "flatpak install --user -y $1"
