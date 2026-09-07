kind: autotools
description: |

    OpenSSH is a complete implementation of the SSH protocol (version 2) for secure remote login, command execution and file transfer.
    It includes a client ssh and server sshd, file transfer utilities scp and sftp as well as tools for key generation (ssh-keygen),
    run-time key storage (ssh-agent) and a number of supporting programs.

    This is a port of OpenBSD's OpenSSH to most Unix-like operating systems, including Linux, OS X and Cygwin.
    Portable OpenSSH polyfills OpenBSD APIs that are not available elsewhere, adds sshd sandboxing for more operating systems
    and includes support for OS-native authentication and auditing (e.g. using PAM).

sources:
- kind: tar
  url: github:openssh/openssh-portable/archive/refs/tags/V_10_5_P1.tar.gz
  ref: 494c0624ed743a4eecc1bdd83d2aab9456bdb4cabc511e00599e71493537e258

depends:
- base.bst
