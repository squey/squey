FROM nvidia/cudagl:10.1-runtime-ubuntu18.04
LABEL maintainer="jean-baptiste+inspector@leonesio.fr"

ARG INSTALL_PATH
ARG INSTALL_MODE
ENV container docker
ENV LC_ALL C
ENV DEBIAN_FRONTEND noninteractive

# Update packages index
ARG APT_PROXY
RUN echo "Acquire::http::Proxy \"$APT_PROXY\";" | tee /etc/apt/apt.conf.d/01proxy
RUN sed -i 's/# deb /deb /g' /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt-get update && \
apt-get -y install apt-utils pkg-config libglvnd-dev libglu1-mesa libglu1 pciutils \
xorg xserver-xorg-input-all xinit xserver-xorg xserver-xorg-video-all xserver-xorg-video-dummy xutils-dev kwin && \
apt-get install -y python3 python3-pip crudini pamtester && \
apt-get install -y nginx ssl-cert && \
apt-get install -y pocl-opencl-icd mesa-utils ocl-icd-libopencl1 clinfo && \
echo "/usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
apt-get -yq install wget patch xterm kdialog && \
apt-get install -y systemd && \
cd /lib/systemd/system/sysinit.target.wants/ && \
ls | grep -v systemd-tmpfiles-setup | xargs rm -f $1 && \
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
COPY download_files.sh env.conf resources/* data/* "${INSTALL_PATH}/"
RUN "${INSTALL_PATH}/download_files.sh" "${INSTALL_MODE}"
RUN chmod g+w "${INSTALL_PATH}"/kwinr*

# Workaround to avoid nvidia drivers to prevent flatpak container to be executed
COPY resources/umount_proc_driver_nvidia.service "${INSTALL_PATH}"
RUN cp "${INSTALL_PATH}/umount_proc_driver_nvidia.service" /lib/systemd/system
RUN systemctl enable umount_proc_driver_nvidia.service

# Install DCV server
RUN find "${INSTALL_PATH}" -name "nice-dcv*.tgz" -exec tar zxvf "{}" -C "/opt" \;
RUN rm -rf /opt/nice-dcv-*/nice-dcv-gl*.deb
RUN dpkg --unpack /opt/nice-dcv-*/nice-dcv-server*.deb
COPY resources/postinst.patch "${INSTALL_PATH}"
RUN bash -c "patch -p1 /var/lib/dpkg/info/nice-dcv-server.postinst < ${INSTALL_PATH}/postinst.patch"
RUN rm -rf /opt/nice-dcv-*/nice-dcv-server*.deb
RUN bash -c "dpkg -i /opt/nice-dcv-*/*.deb"; exit 0
RUN apt-get --fix-broken install -y
RUN systemctl enable dcvserver.service

# Configure DCV license server
ARG DCV_LICENSE_SERVER
RUN sed -i "s/#license-file = \"\"/license-file = \"${DCV_LICENSE_SERVER}\"/g" /etc/dcv/dcv.conf

# Configure INENDI Inspector license server
ARG INSPECTOR_LICENSE_SERVER
RUN crudini --set "${INSTALL_PATH}/../inspector.conf" "" "license_server" "${INSPECTOR_LICENSE_SERVER}"

# Configure DCV login handling
COPY resources/dcvsessioninit /etc/dcv/
RUN chmod +x /etc/dcv/dcvsessioninit
COPY resources/dcvlogin "${INSTALL_PATH}/dcvlogin"
RUN chmod +x "${INSTALL_PATH}/dcvlogin/dcvlogin.py"
RUN pip3 install --system --requirement "${INSTALL_PATH}/dcvlogin/requirements.txt"
RUN sed -i "s|#auth-token-verifier=\(.*\)|auth-token-verifier=\"http://127.0.0.1:8444\"|g" /etc/dcv/dcv.conf
RUN ln -s "${INSTALL_PATH}/dcvlogin/dcvlogin.service" /lib/systemd/system || true
RUN rm -rf /etc/nginx/sites-available/default || true
RUN ln -s "${INSTALL_PATH}/dcvlogin/nginx_proxy" /etc/nginx/sites-available/default || true
RUN systemctl enable nginx
RUN systemctl enable dcvlogin
RUN systemctl enable dcvsimpleextauth

# Configure INENDI Inspector
COPY resources/inspector_loop.sh "${INSTALL_PATH}/"
RUN chmod +x "${INSTALL_PATH}/inspector_loop.sh"
RUN chmod +x "${INSTALL_PATH}/cuda-devices"
RUN mkdir -p /srv/tmp-inspector
RUN useradd --system inendi
RUN chown inendi: /srv/tmp-inspector
RUN chmod g+w /srv/tmp-inspector

# Configure KWin window manager
RUN mkdir /etc/kde5
RUN ln -s "${INSTALL_PATH}/kwinrc" /etc/kde5 || true
RUN ln -s "${INSTALL_PATH}/kwinrulesrc" /etc/kde5 || true
RUN chown inendi: /etc/kde5
RUN chmod g+w /etc/kde5 -R

# Cleanup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

VOLUME [ "/sys/fs/cgroup" ]
CMD [ "/lib/systemd/systemd" ]
