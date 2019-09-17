#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${DIR}/config.env"
source "${DIR}/.config.env"

if [ -z "${USER_NAME}" ]
then
    echo -n "username: " 
    read USER_NAME
fi

groupadd "${USER_NAME}"
useradd "${USER_NAME}" -m -g "${USER_NAME}"
chsh -s /bin/bash "${USER_NAME}"
usermod -a -G inendi,adm,video,plugdev "${USER_NAME}"

if [ -z "${USER_PASSWORD}" ]
then
    passwd "${USER_NAME}"
else
    echo "${USER_NAME}:${USER_PASSWORD}" | chpasswd
fi

sed -i "s/User=\([[:alnum:]]*\)/User=$USER_NAME/g" /lib/systemd/system/dcvsession.service
sed -i "s/--user=\([[:alnum:]]*\)/--user=$USER_NAME/g" /lib/systemd/system/dcvsession.service
sed -i "s/--owner=\([[:alnum:]]*\)/--owner=$USER_NAME/g" /lib/systemd/system/dcvsession.service

# Configure Window manager (KWin) 
mkdir -p /home/${USER_NAME}/.config
cp "${INSTALL_PATH}/kwinrc" /home/${USER_NAME}/.config
cp "${INSTALL_PATH}/kwinrulesrc" /home/${USER_NAME}/.config
chown -R ${USER_NAME}: /home/${USER_NAME}/.config
