FROM nvidia/cudagl:10.1-runtime-ubuntu18.04
LABEL maintainer="jean-baptiste.leonesio@esi-group.com"

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
apt-get -y install apt-utils pkg-config libglvnd-dev mesa-utils libglu1-mesa libglu1 ocl-icd-libopencl1 pciutils xfonts-base \
lightdm xorg xserver-xorg-input-all xinit xserver-xorg xserver-xorg-video-all xserver-xorg-video-dummy xutils-dev kwin && \
apt-get install -y pocl-opencl-icd mesa-utils ocl-icd-libopencl1 clinfo && \
echo "/usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
apt-get -yq install awscli wget patch xxd xterm && \
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

# Configure dbus TODO : check if necessary
RUN dbus-uuidgen > /var/lib/dbus/machine-id
RUN mkdir -p /var/run/dbus

# Allow non-privileged user namespaces and increase the maximum number of memory map areas
RUN echo "kernel.unprivileged_userns_clone = 1" > /etc/sysctl.d/00-local-userns.conf
RUN echo "vm.max_map_count=1966080" > /etc/sysctl.d/00-local-max_map_count.conf
    
# Download/install data
RUN mkdir -p "${INSTALL_PATH}"
COPY download_files.sh config.env .config.env data/* "${INSTALL_PATH}/"
RUN "${INSTALL_PATH}/download_files.sh" "${INSTALL_MODE}"

# Workaround to avoid nvidia drivers to prevent flatpak container to be executed
COPY umount_proc_driver_nvidia.service "${INSTALL_PATH}"
RUN cp "${INSTALL_PATH}/umount_proc_driver_nvidia.service" /lib/systemd/system
RUN systemctl enable umount_proc_driver_nvidia.service

# Install DCV server
RUN find "${INSTALL_PATH}" -name "nice-dcv*.tgz" -exec tar zxvf "{}" -C "/opt" \;
RUN dpkg --unpack /opt/nice-dcv-*/nice-dcv-server*.deb
COPY postinst.patch "${INSTALL_PATH}"
RUN bash -c "patch -p1 /var/lib/dpkg/info/nice-dcv-server.postinst < ${INSTALL_PATH}/postinst.patch"
RUN rm -rf /opt/nice-dcv-*/nice-dcv-server*.deb
RUN bash -c "dpkg -i /opt/nice-dcv-*/*.deb"; exit 0
RUN apt-get --fix-broken install -y

# Configure DCV license server
ARG DCV_LICENSE_SERVER
RUN sed -i "s/#license-file = \"\"/license-file = \"${DCV_LICENSE_SERVER}\"/g" /etc/dcv/dcv.conf

# Configure DCV OpenGL support
RUN mkdir -p /var/lib/dcv-gl/flatpak
RUN cp /usr/lib/x86_64-linux-gnu/dcv-gl/libGL* /var/lib/dcv-gl/flatpak
RUN chmod +x "${INSTALL_PATH}/patch_strings_in_file.sh"
RUN "${INSTALL_PATH}/patch_strings_in_file.sh" "/var/lib/dcv-gl/flatpak/libGL_WRAPPER.so.1.0.0" "%s/dcv-gl/libGL_DCV.so" "libGL_DCV.so"

# Configure DCV session
COPY dcvsessioninit /etc/dcv/dcvsessioninit
RUN chmod +x /etc/dcv/dcvsessioninit
RUN cp "${INSTALL_PATH}/dcvsession.service" /lib/systemd/system
RUN systemctl enable dcvserver.service
RUN systemctl enable dcvsession.service

# Configure INENDI Inspector
COPY inspector_loop.sh "${INSTALL_PATH}/inspector_loop.sh"
RUN chmod +x "${INSTALL_PATH}/inspector_loop.sh"
RUN chmod +x "${INSTALL_PATH}"/cuda-devices
RUN mkdir -p /srv/tmp-inspector
RUN useradd --system inendi
RUN chown inendi: /srv/tmp-inspector
RUN chmod g+w /srv/tmp-inspector

# Cleanup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

VOLUME [ "/sys/fs/cgroup" ]
CMD [ "/lib/systemd/systemd" ]
