FROM docker.io/nvidia/cudagl:11.4.2-runtime-ubuntu20.04
LABEL maintainer="jean-baptiste+squey@leonesio.fr"

ARG INSTALL_PATH
ARG INSTALL_MODE
ENV container docker
ENV LC_ALL C
ENV DEBIAN_FRONTEND noninteractive

# Update packages index
ARG APT_PROXY
RUN echo "Acquire::http::Proxy \"$APT_PROXY\";" | tee /etc/apt/apt.conf.d/01proxy
RUN cat /etc/apt/apt.conf.d/01proxy
RUN sed -i 's/# deb /deb /g' /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv A4B469963BF863CC
RUN apt-get update && apt-get -y dist-upgrade && \
apt-get install -y --no-install-recommends apt-utils pkg-config libglvnd-dev libglu1-mesa \ 
libglu1 pciutils xorg xserver-xorg-input-all xinit xserver-xorg xserver-xorg-video-all \
xserver-xorg-video-dummy xutils-dev kwin software-properties-common software-properties-common python3 python3-pip \
crudini pamtester nginx ssl-cert pocl-opencl-icd mesa-utils ocl-icd-libopencl1 clinfo wget patch \
xterm kdialog systemd libxcb-xtest0 libcairo2 libcrack2 libgstreamer-plugins-base1.0-0 libgtk-3-0 \
v4l2loopback-dkms libvdpau1 libva-drm2 qdbus-qt5 sudo && \
add-apt-repository -y ppa:flatpak/stable && apt-get update && apt-get install -y flatpak && \
echo "/usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
cd /lib/systemd/system/sysinit.target.wants/ && \
ls | grep -v systemd-tmpfiles-setup | xargs rm -f $1 && \
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
rm -f /lib/systemd/system/multi-user.target.wants/* \
    /etc/systemd/system/*.wants/* \
    /lib/systemd/system/local-fs.target.wants/* \
    /lib/systemd/system/sockets.target.wants/*udev* \
    /lib/systemd/system/sockets.target.wants/*initctl* \
    /lib/systemd/system/basic.target.wants/* \
    /lib/systemd/system/anaconda.target.wants/* \
    /lib/systemd/system/plymouth* \
    /lib/systemd/system/systemd-update-utmp*

# Allow non-privileged user namespaces and increase the maximum number of memory map areas
RUN echo "kernel.unprivileged_userns_clone = 1" > /etc/sysctl.d/00-local-userns.conf
RUN echo "vm.max_map_count=1966080" > /etc/sysctl.d/00-local-max_map_count.conf

# Disable coredumpctl core dump handling
RUN echo 'kernel.core_pattern=|/bin/false' > /etc/sysctl.d/50-coredump.conf
    
# Download/install data
RUN mkdir -p "${INSTALL_PATH}"
COPY download_files.sh env.conf resources/. data/. "${INSTALL_PATH}/"
RUN "${INSTALL_PATH}/download_files.sh" "${INSTALL_MODE}"
RUN chmod go+w "${INSTALL_PATH}"/kwinr*

# Workaround to avoid nvidia drivers to prevent flatpak container to be executed
RUN cp "${INSTALL_PATH}/umount_proc_driver_nvidia.service" /lib/systemd/system
RUN systemctl enable umount_proc_driver_nvidia.service

# Install Squey
RUN chmod +x "${INSTALL_PATH}/install_files.sh"
RUN "${INSTALL_PATH}/install_files.sh" "${INSTALL_MODE}"

# Configure authentification
RUN chmod +x "${INSTALL_PATH}/configure_auth.sh"
RUN "${INSTALL_PATH}/configure_auth.sh"

# Configure Squey
RUN echo "ALL ALL=(ALL:ALL) NOPASSWD: /usr/bin/flatpak" | tee /etc/sudoers.d/flatpak
RUN chmod +x "${INSTALL_PATH}/squey_loop.sh"
RUN chmod +x "${INSTALL_PATH}/cuda-devices"
RUN mkdir -p /srv/tmp-squey
RUN chgrp squey /srv/tmp-squey
RUN chmod g+w /srv/tmp-squey

# Install DCV server
RUN chmod +x "${INSTALL_PATH}/install_dcv_server.sh"
RUN  ${INSTALL_PATH}/install_dcv_server.sh

# Configure DCV SSL certificate
RUN chmod +x "${INSTALL_PATH}/configure_ssl.sh"
RUN  ${INSTALL_PATH}/configure_ssl.sh

# Configure DCV license server
ARG DCV_LICENSE_SERVER
RUN sed -i "s/#license-file = \"\"/license-file = \"${DCV_LICENSE_SERVER}\"/g" /etc/dcv/dcv.conf

# Configure DCV login handling
RUN rm -rf /etc/dcv/dcvsessioninit && ln -s "${INSTALL_PATH}/dcvsessioninit" /etc/dcv/
RUN chmod +x /etc/dcv/dcvsessioninit
RUN chmod +x "${INSTALL_PATH}/dcvlogin/dcvlogin.py"
RUN pip3 install --system --requirement "${INSTALL_PATH}/dcvlogin/requirements.txt"
RUN sed -i "s|#auth-token-verifier=\(.*\)|auth-token-verifier=\"http://127.0.0.1:8444\"|g" /etc/dcv/dcv.conf
RUN ln -s "${INSTALL_PATH}/dcvlogin/dcvlogin.service" /lib/systemd/system || true
RUN rm -rf /etc/nginx/sites-available/default || true
RUN ln -s "${INSTALL_PATH}/dcvlogin/nginx_proxy" /etc/nginx/sites-available/default || true

# Configure KWin window manager
RUN mkdir /etc/kde5
RUN ln -s "${INSTALL_PATH}/kwinrc" /etc/kde5 || true
RUN ln -s "${INSTALL_PATH}/kwinrulesrc" /etc/kde5 || true
RUN chown squey: /etc/kde5
RUN chmod g+w /etc/kde5 -R

# Configure services to run uppon container launch
RUN cp /usr/lib/systemd/system/nginx.service /etc/systemd/system
RUN sed -i 's/After=network.target/After=dcvlogin.service/' /etc/systemd/system/nginx.service
RUN systemctl enable dcvserver dcvlogin dcvsimpleextauth nginx


VOLUME [ "/sys/fs/cgroup" ]
CMD [ "/lib/systemd/systemd" ]
