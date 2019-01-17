#!/bin/bash

NVIDIA_FTP_URL="http://download.nvidia.com/XFree86"

_dir=`dirname $0`

log() {
    echo "$1"
}

_system=
_os_version=

check_versions() {
    local min_version=$1
    local check_version=$2
    local ver=$(printf "%s\n%s" "${check_version}" "${min_version}" | sort -n | head -1)
    echo "${ver}"
}

disable_selinux() {
    sed -i 's/SELINUX\=enforcing/SELINUX\=disabled/g' /etc/selinux/config
}

disable_wayland() {
    sed -i 's/#WaylandEnable/WaylandEnable/g' /etc/gdm/custom.conf
}

check_nvidia_card() {
    lspci 2>/dev/null | grep -i nvidia > /dev/null
    # TODO: maybe also grep for quadro/grid/tesla and warn if it is a consumer card?
    if [ "$?" = "0" ]; then
        return 0
    else
        return 1
    fi
}

check_nouveau() {
    # The nvidia driver itself checks for nouveau, but usually fails to
    # disable it: if we detect that nvidia alreay installed the blacklist
    # but the module is still there, then print some advice for the user
    # instead of looping through the nvidia installer again (we do not
    # want to fiddle with grub ourselves)
    lsmod | grep -i nouveau > /dev/null
    if [ $? = "0" -a -f /etc/modprobe.d/nvidia-installer-disable-nouveau.conf ]; then
        return 1
    fi

    return 0
}

disable_nouveau() {
    # from http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
    grep nouveau /etc/modprobe.d/blacklist.conf > /dev/null
    if [ "$?" = "1" ]; then
        echo "blacklist vga16fb" >> /etc/modprobe.d/blacklist.conf
        echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf
        echo "blacklist rivafb" >> /etc/modprobe.d/blacklist.conf
        echo "blacklist nvidiafb" >> /etc/modprobe.d/blacklist.conf
        echo "blacklist rivatv" >> /etc/modprobe.d/blacklist.conf
    fi

    if [ -f /etc/default/grub ]; then
        if grep GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub >/dev/null; then
            sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT=".*\)"/\1 modprobe.blacklist=nouveau"/' /etc/default/grub
        else
            echo 'GRUB_CMDLINE_LINUX_DEFAULT="modprobe.blacklist=nouveau"' >> /etc/default/grub
        fi
        grub-mkconfig -o /boot/grub/grub.cfg
    else
        sed -i 's/\(kernel .*\)/\1 rdblacklist=nouveau nouveau.modeset=0/' /boot/grub/grub.conf
    fi
}

download_nvidia_driver() {
    local arch_path="Linux-x86_64"
    local latest_path="${NVIDIA_FTP_URL}/${arch_path}/latest.txt"

    log "Retrieving latest nvidia version"
    local version=$(curl --connect-timeout 5 "${latest_path}"  2> /dev/null | cut -d' ' -f1)

    if [ -n "${version}" ]; then
        local driver_name="NVIDIA-${arch_path}-${version}.run"
        local driver_path="${NVIDIA_FTP_URL}/${arch_path}/${version}/${driver_name}"

        log "Retrieving NVIDIA driver: ${driver_path}"
        curl -# -o "/opt/dcv-install/${driver_name}" "${driver_path}"
        if [ "$?" = "0" ]; then
            ln -s "/opt/dcv-install/${driver_name}" "/opt/dcv-install/NVIDIA-${arch_path}.run"
        else
            log "Failed to download latest nvidia driver"
        fi
    else
        log "Failed to find latest nvidia version"
    fi
}

download_nvidia_driver_for_g3() {
    local arch_path="Linux-x86_64"
    local driver_name=""

    log "Retrieving NVIDIA driver from S3"

    driver_name=`aws s3 ls s3://ec2-linux-nvidia-drivers/ | awk '{print $4}' | grep NVIDIA-${arch_path}-.*-grid.run`

    log `echo ${driver_name}`
    if [ -n ${driver_name} ]; then
        aws s3 cp "s3://ec2-linux-nvidia-drivers/${driver_name}" /opt/dcv-install/

        if [ "$?" = "0" ]; then
            log "NVIDIA driver downloaded"
            ln -s "/opt/dcv-install/${driver_name}" "/opt/dcv-install/NVIDIA-${arch_path}-g3.run"
        else
            log "Failed to download nvidia driver from S3"
        fi
    else
        log "Unable to find nvidia driver on S3"
    fi
}

install_packages() {
    _version=$1
    _install_dcv_gl=$2

    dcv_tgz_url="https://d1uj6qtbmh3dt5.cloudfront.net/server/"
    dcv_tgz="nice-dcv-2017.2-6182-ubuntu1804.tgz"

    curl "${dcv_tgz_url}${dcv_tgz}" -o "${_dir}/${dcv_tgz}"
    tar zxvf "${_dir}/${dcv_tgz}" -C "${_dir}/"
    dpkg -i "${_dir}"/nice-dcv-*/*.deb
    apt --fix-broken install -y
}

if [ `id -u` -ne 0 ]; then
    log "You must be root to deploy dcv."
    exit 1
fi

if [ ! -f "${_dir}/conf.sh" ]; then
    log "Conf file does not exists. Exiting..."
    exit 1
fi

. "${_dir}/conf.sh"

apt-get -y update
export DEBIAN_FRONTEND=noninteractive ; apt-get dist-upgrade -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" --force-yes
apt-get -y install linux-headers-generic gcc make dkms pkg-config libglvnd-dev mesa-utils libglu1-mesa libglu1 ocl-icd-libopencl1 pciutils xfonts-base lightdm xorg xserver-xorg-input-all xinit xserver-xorg xserver-xorg-video-all xserver-xorg-video-dummy xutils-dev
apt-get -y install wget vim screen kwin xterm

check_nvidia_card
has_nvidia_card=$?

# Note handle firewalld after we install the workstation
ufw disable
#disable_selinux

install_packages "ubuntu1804" "${has_nvidia_card}"

# Enable dcvserver
systemctl enable dcvserver.service

if [ "${has_nvidia_card}" = "0" ]; then
    check_nouveau
    if [ "$?" = "0" ]; then
        disable_nouveau
    fi

    download_nvidia_driver
    download_nvidia_driver_for_g3
else
    aws s3 cp "s3://${dep_bucket}/xorg.conf" "${_dir}/xorg.conf"
    mv -fb "${_dir}/xorg.conf" /etc/X11/xorg.conf
fi

# add user
groupadd ${user_name}
useradd ${user_name} -m -g ${user_name}
chsh -s /bin/bash "${user_name}"
echo "${user_name}:${user_pass}" | chpasswd
usermod -a -G adm,dialout,cdrom,floppy,sudo,audio,dip,video,plugdev,lxd,netdev ${user_name}
mkdir "/home/${user_name}/.ssh"
cp "/home/ubuntu/.ssh/authorized_keys" "/home/${user_name}/.ssh"
chown "${user_name}" "/home/${user_name}/.ssh" -R
chmod 700 "/home/${user_name}/.ssh"
chmod 600 "/home/${user_name}/.ssh/authorized_keys"
echo "Created user ${user_name}"

# remove default user
deluser --remove-home ubuntu

# create startup script
INSPECTOR_BUCKET="inendi-inspector"
aws s3 cp "s3://${INSPECTOR_BUCKET}/dcvserverinit" "${_dir}/dcvserverinit"
cp "${_dir}/dcvserverinit" /etc/init.d/
chmod +x /etc/init.d/dcvserverinit
update-rc.d dcvserverinit defaults
echo "Enabled dcvserverinit service"

# write the config file
cat > /etc/dcv/dcv.conf << EOF
[session-management]
create-session=true
owner="${user_name}"
EOF
echo "Created configuration file"

echo "Installing Inspector"

INSPECTOR_INSTALL_DIR="/opt/inspector-install"
mkdir -p "${INSPECTOR_INSTALL_DIR}"
cd "${INSPECTOR_INSTALL_DIR}"
aws s3 cp "s3://${INSPECTOR_BUCKET}/inspector-install.sh" "${INSPECTOR_INSTALL_DIR}/install.sh"

set -x

cd "${INSPECTOR_INSTALL_DIR}"
. ./install.sh > install.log 2>&1

echo "Rebooting..."

reboot

# ex:set ts=4 et:
