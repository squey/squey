#!/bin/sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -f "${DIR}/ssh_host_rsa_key" ] || [ ! -f "${DIR}/ssh_host_dsa_key" ]; then
    /app/bin/ssh-keygen -f "${DIR}/ssh_host_rsa_key" -N '' -t rsa > /dev/null
    /app/bin/ssh-keygen -f "${DIR}/ssh_host_dsa_key" -N '' -t dsa > /dev/null
fi

cat << EOF > "${DIR}/sshd_config"
# Do not edit this file as it will be overriden by run_ssh_server.sh script !
Port 6666
HostKey "${DIR}/ssh_host_rsa_key"
HostKey "${DIR}/ssh_host_dsa_key"
PidFile "${DIR}/sshd.pid"
AllowAgentForwarding yes
AllowTcpForwarding yes
X11Forwarding yes
X11DisplayOffset 10
X11UseLocalhost no
AcceptEnv XDG_RUNTIME_DIR
EOF

/app/sbin/sshd -f "${DIR}/sshd_config" -E "${DIR}/sshd.log"
/app/bin/waypipe --socket /tmp/squey-waypipe-socket-server server