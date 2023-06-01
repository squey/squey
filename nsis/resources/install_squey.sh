#!/bin/sh

if [ $# -ne 1 ]; then
	echo "usage : `basename $0` <flatpakref URL>"
    exit 1
fi

sed 's|alpine/.*/|alpine/latest-stable/|' -i /etc/apk/repositories
apk add -U flatpak
addgroup -S squey && adduser -D squey -G squey
mkdir -p /srv/tmp-squey
chown squey: -R /srv/tmp-squey
su - squey -s "/bin/ash" -c "flatpak install --user -y --no-related $1"