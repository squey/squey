#!/bin/sh

if [ $# -ne 1 ]; then
	echo "usage : `basename $0` <flatpakref URL>"
    exit 1
fi

sed 's|alpine/.*/|alpine/latest-stable/|' -i /etc/apk/repositories
apk add -U flatpak
addgroup -S inendi && adduser -D inendi -G inendi
mkdir -p /srv/tmp-inspector
chown inendi: -R /srv/tmp-inspector
su - inendi -s "/bin/ash" -c "flatpak install --user -y $1"