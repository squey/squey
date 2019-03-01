#!/bin/bash

_dir="${INSPECTOR_INSTALL_DIR}"

# Get needed files from S3 bucket
aws s3 cp "s3://${INSPECTOR_BUCKET}/cuda-devices" "${_dir}/cuda-devices"
aws s3 cp "s3://${INSPECTOR_BUCKET}/dcvsession.service" "${_dir}/dcvsession.service"
aws s3 cp "s3://${INSPECTOR_BUCKET}/nginx.conf" "${_dir}/nginx.conf"
aws s3 cp "s3://${INSPECTOR_BUCKET}/patch_strings_in_file.sh" "${_dir}/patch_strings_in_file.sh"
aws s3 cp "s3://${INSPECTOR_BUCKET}/dcvsessioninit" "${_dir}/dcvsessioninit"
aws s3 cp "s3://${INSPECTOR_BUCKET}/kwinrc" "${_dir}/kwinrc"
aws s3 cp "s3://${INSPECTOR_BUCKET}/kwinrulesrc" "${_dir}/kwinrulesrc"

# Install Flatpak
add-apt-repository -y ppa:alexlarsson/flatpak
apt-get -y update
apt-get install -y flatpak xdg-desktop-portal

# Install Inspector
if [ "${product}" = "INENDI Inspector" ]; then
    sudo -u ${user_name} -H flatpak install --user -y https://repo.esi-inendi.com/inendi-inspector.flatpakref
    PACKAGE_NAME="com.esi_inendi.Inspector"
elif [ "${product}" = "PCAP Inspector" ]; then
    sudo -u ${user_name} -H flatpak install --user -y https://pcap-inspector.com/inspector.flatpakref
    PACKAGE_NAME="com.pcap_inspector.Inspector"
fi
echo "flatpak update --user -y; while true; do flatpak run --filesystem=/var/lib/dcv-gl/flatpak --share=ipc $PACKAGE_NAME; done" > "${_dir}/inspector_loop.sh"
mkdir -p /srv/tmp-inspector
chown ${user_name}: /srv/tmp-inspector

# Configure KWin session
mkdir /home/${user_name}/.config
ln -s "${_dir}/kwinrc" /home/${user_name}/.config
ln -s "${_dir}/kwinrulesrc" /home/${user_name}/.config
chown -R ${user_name}: /home/${user_name}/.config
chmod +x "${_dir}/inspector_loop.sh"

#Configure DCV OpenGL
chmod +x "${_dir}/cuda-devices"
echo "
[display/linux]
gl-displays = [':0', ':1' ]" >> "/etc/dcv/dcv.conf"
mkdir -p /var/lib/dcv-gl/flatpak
cp /usr/lib/x86_64-linux-gnu/dcv-gl/libGL* /var/lib/dcv-gl/flatpak
chmod +x "${_dir}/patch_strings_in_file.sh"
"${_dir}/patch_strings_in_file.sh" "/var/lib/dcv-gl/flatpak/libGL_WRAPPER.so.1.0.0" "%s/dcv-gl/libGL_DCV.so" "libGL_DCV.so"

# Change DCV default configuration
DCV_CONFIG_PATH=/etc/dcv/dcv.conf
sed -i 's/create-session=true/create-session=false/g' "${DCV_CONFIG_PATH}"
systemctl restart dcvserver

# Configure DCV session to launch at startup
cp "${_dir}/dcvsession.service" /lib/systemd/system
cp "${_dir}/dcvsessioninit" "/etc/dcv"
chmod +x "/etc/dcv/dcvsessioninit"
systemctl enable dcvsession
systemctl start dcvsession
systemctl restart dcvserver

# Configure nginx reverse proxy
apt-get install -y nginx
rm -rf "/etc/nginx/sites-available/default"
ln -s "${_dir}/nginx.conf" "/etc/nginx/sites-available/default"
systemctl enable nginx
systemctl restart nginx

# Increase the maximum number of memory map areas
echo 'vm.max_map_count=1966080' >> /etc/sysctl.conf
sysctl -p
