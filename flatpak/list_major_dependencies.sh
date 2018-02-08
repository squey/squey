#!/bin/sh

function get_redhat_versions() {
    VERSION=$1
    case $VERSION in
        5 )
            DIRNAME=${VERSION}/os/i386/CentOS/ ;;
        6 )
            DIRNAME=${VERSION}/os/x86_64/Packages/ ;;
        7 )
            DIRNAME=${VERSION}/os/x86_64/Packages/ ;;
    esac

    local url="http://mirror.centos.org/centos/$DIRNAME"
    
    local pre_parse=$(mktemp --suffix=.awk)
    cat >$pre_parse <<EOF
BEGIN {FS="-"}
// {
    pkg=\$1
    for (i = 2; i < NF-1; i++)
        pkg=(pkg "-" \$i)
    
    print \$1 " " \$(NF-1)
};
EOF
    local parse=$(mktemp --suffix=.awk)
    cat >$parse <<EOF
BEGIN {
    PROCINFO["sorted_in"] = "@ind_str_asc"
}
    
\$1 ~ /^gcc[0-9]*/ {GCC[\$2] = \$2};
\$1 == "glibc" {GLIBC=\$2};
\$1 == "zlib" {ZLIB=\$2};
\$1 == "libxml2" {XML2=\$2};
\$1 == "pcre" {PCRE=\$2};
\$1 == "nss" {NSS=\$2};
\$1 == "curl" {CURL=\$2};
END {
    for (gcc_version in GCC)
      GCC_str=(GCC_str " " gcc_version)

    printf "RedHat %s;%s;%s;%s;%s;%s;%s;%s\n",VERSION,GCC_str,GLIBC,ZLIB,XML2,PCRE,NSS,CURL
}
EOF

    curl -s --list-only $url |sed -n 's/.*\"\([a-zA-Z0-9\.\_\-]\+\)\.rpm\".*/\1/p' |awk -f $pre_parse |awk -f $parse -vVERSION=$VERSION
    [ -f $pre_parse ] && rm $pre_parse
    [ -f $parse ] && rm $parse
}

function get_fedora_versions() {
    VERSION=$1

    local url="ftp://fedora.mirrors.ovh.net/linux/releases/${VERSION}/Workstation/source/SRPMS"
    
    # Create all urls list
    local all_urls=""
    for l in {a..z} ; do
        all_urls="$all_urls $url/$l/"
    done
    
    local pre_parse=$(mktemp --suffix=.awk)
    cat >$pre_parse <<EOF
BEGIN {FS="-"}
{
    pkg=\$1
    for (i = 2; i < NF-1; i++)
        pkg=(pkg "-" \$i)
    
    print \$1 " " \$(NF-1)
};
EOF
    
    local parse=$(mktemp --suffix=.awk)
    cat >$parse <<EOF
BEGIN { PROCINFO["sorted_in"] = "@ind_str_asc" }
    
\$1 == "gcc" {GCC = \$2};
\$1 == "glibc" {GLIBC=\$2};
\$1 == "zlib" {ZLIB=\$2};
\$1 == "libxml2" {XML2=\$2};
\$1 == "pcre" {PCRE=\$2};
\$1 == "nss" {NSS=\$2};
\$1 == "curl" {CURL=\$2};
END {
    printf "Fedora %s;%s;%s;%s;%s;%s;%s;%s\n",VERSION,GCC,GLIBC,ZLIB,XML2,PCRE,NSS,CURL}
EOF
    
    curl -s --list-only $all_urls |awk -f $pre_parse |awk -f $parse -vVERSION=$VERSION
    [ -f $pre_parse ] && rm $pre_parse
    [ -f $parse ] && rm $parse
}

function get_opensuse_versions() {
    VERSION=$1

    local url="ftp://opensuse.mirrors.ovh.net/opensuse/distribution/${VERSION}/repo/oss/suse/x86_64/"
    
    local pre_parse=$(mktemp --suffix=.awk)
    cat >$pre_parse <<EOF
BEGIN {FS="-"}
{
    pkg=\$1
    for (i = 2; i < NF-1; i++)
        pkg=(pkg "-" \$i)
    
    print \$1 " " \$(NF-1)
};
EOF
    
    local parse=$(mktemp --suffix=.awk)
    cat >$parse <<EOF
BEGIN { PROCINFO["sorted_in"] = "@ind_str_asc" }
    
\$1 == "gcc" {GCC = \$2};
\$1 == "glibc" {GLIBC=\$2};
\$1 == "zlib" {ZLIB=\$2};
\$1 == "libxml2" {XML2=\$2};
\$1 == "pcre" {PCRE=\$2};
\$1 == "nss" {NSS=\$2};
\$1 == "curl" {CURL=\$2};
END {
    printf "OpenSuse %s;%s;%s;%s;%s;%s;%s;%s\n",VERSION,GCC,GLIBC,ZLIB,XML2,PCRE,NSS,CURL}
EOF
    
    curl -s --list-only $url |awk -f $pre_parse |awk -f $parse -vVERSION=$VERSION
    [ -f $pre_parse ] && rm $pre_parse
    [ -f $parse ] && rm $parse
}

function get_debian_versions() {
    VERSION=$1
    
    local url="ftp://ftp2.fr.debian.org/debian/dists/$VERSION/main/source/Sources.gz"
    
    local pre_parse=$(mktemp --suffix=.awk)
    cat >$pre_parse <<EOF
/^Package:/ {pkg = \$2};
/^Version:/{
    sub(/^[0-9]*:/, "", \$2)
    split(\$2,version, "-");
    print pkg " " version[1];
}
EOF
    local parse=$(mktemp --suffix=.awk)
    cat >$parse <<EOF
BEGIN { PROCINFO["sorted_in"] = "@ind_str_asc" }

\$1 ~ /^gcc-[0-9].[0-9]\$/ {GCC[\$2] = \$2};
\$1 == "eglibc" {GLIBC=\$2}
\$1 == "zlib" {ZLIB=\$2}
\$1 == "libxml2" {XML2=\$2}
\$1 == "pcre3" {PCRE=\$2}
\$1 == "nss" {NSS=\$2}
\$1 == "curl" {CURL=\$2}
END {
    for (gcc_version in GCC)
      GCC_str=(GCC_str " " gcc_version)

    printf "Debian %s;%s;%s;%s;%s;%s;%s;%s\n",VERSION,GCC_str,GLIBC,ZLIB,XML2,PCRE,NSS,CURL
}
EOF

    curl -s $url |gunzip |awk -f $pre_parse |awk -f $parse -vVERSION=$VERSION
    [ -f $pre_parse ] && rm $pre_parse
    [ -f $parse ] && rm $parse
}

function get_ubuntu_versions() {
    VERSION=$1
    
    local url="ftp://ftp.ubuntu.com/ubuntu/dists/$VERSION/main/source/Sources.gz"
    
    local pre_parse=$(mktemp --suffix=.awk)
    cat >$pre_parse <<EOF
/^Package:/ {pkg = \$2};
/^Version:/{
    sub(/^[0-9]*:/, "", \$2)
    split(\$2,version, "-");
    print pkg " " version[1];
}
EOF
    local parse=$(mktemp --suffix=.awk)
    cat >$parse <<EOF
BEGIN { PROCINFO["sorted_in"] = "@ind_str_asc" }

\$1 ~ /^gcc-[0-9].[0-9]\$/ {GCC[\$2] = \$2};
\$1 == "eglibc" {GLIBC=\$2}
\$1 == "zlib" {ZLIB=\$2}
\$1 == "libxml2" {XML2=\$2}
\$1 == "pcre3" {PCRE=\$2}
\$1 == "nss" {NSS=\$2}
\$1 == "curl" {CURL=\$2}
END {
    for (gcc_version in GCC)
      GCC_str=(GCC_str " " gcc_version)

    printf "Ubuntu %s;%s;%s;%s;%s;%s;%s;%s\n",VERSION,GCC_str,GLIBC,ZLIB,XML2,PCRE,NSS,CURL
}
EOF

    curl -s $url |gunzip |awk -f $pre_parse |awk -f $parse -vVERSION=$VERSION
    [ -f $pre_parse ] && rm $pre_parse
    [ -f $parse ] && rm $parse
}

echo "" "gcc" "glibc" "zlib" "libxml2" "pcre" "nss" "curl" |tr ' ' ';'
get_redhat_versions 5
get_redhat_versions 6
get_redhat_versions 7
get_fedora_versions 22
get_fedora_versions 23
get_debian_versions stable
get_debian_versions testing
get_ubuntu_versions lucid
get_ubuntu_versions precise
get_ubuntu_versions trusty
get_opensuse_versions 12.3
get_opensuse_versions 13.1

