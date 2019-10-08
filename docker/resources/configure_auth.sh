#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${DIR}/env.conf"

if [ ! -z "${LDAP_URI}" ] # Configure PAM LDAP authentication if specified
then
    export DEBIAN_FRONTEND=noninteractive
    apt update && apt -y install libnss-ldap libpam-ldap ldap-utils nscd && rm -rf /var/lib/apt/lists/*
    sed -i "s|^uri \(.*\)|uri ${LDAP_URI}|g" /etc/ldap.conf
    sed -i "s/^base \(.*\)/base ${LDAP_BASE_DN}/g" /etc/ldap.conf
    sed -i "s/^rootbinddn \(.*\)/#rootbinddn \1/g" /etc/ldap.conf
    if [ ! -z "${LDAP_ADMIN_DN}" ]
    then
        sed -i "s/^#binddn \(.*\)/binddn ${LDAP_BIND_DN}/g" /etc/ldap.conf
        sed -i "s/^#bindpw \(.*\)/bindpw ${LDAP_BIND_PASSWORD}/g" /etc/ldap.conf
    fi
    echo "nss_override_attribute_value gidNumber `id -u inendi`" >> /etc/ldap.conf
    auth-client-config -t nss -p lac_ldap
    pam-auth-update --package --enable ldap --enable unix --enable systemd
    systemctl enable nscd
    systemctl restart nscd
fi

crudini --set /etc/adduser.conf '' DIR_MODE 0700
if [ ! -z "${USER_NAME}" ] # Create local user if specified
then
    groupadd "${USER_NAME}"
    useradd "${USER_NAME}" -m -g "${USER_NAME}"
    if [ $? -eq 0 ]
    then
        chsh -s /bin/bash "${USER_NAME}"
        usermod -a -G inendi,adm,video,plugdev "${USER_NAME}"

        if [ -z "${USER_PASSWORD}" ]
        then
            passwd "${USER_NAME}"
        else
            echo "${USER_NAME}:${USER_PASSWORD}" | chpasswd
        fi
    fi
fi
