#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

add_subdirectory(normalize)
add_subdirectory(input_types)
add_subdirectory(sources)

################################################
# Ressource declaration
################################################

SET(NORMALIZE_PCAP_FORMAT_FILES
normalize-helpers/pcap/default.format
normalize-helpers/pcap/local-192-168-src.format
)

SET(NORMALIZE_PERL_FORMAT_FILES
normalize-helpers/perl/IronPort.format
normalize-helpers/perl/IronPort.pl
)

SET(NORMALIZE_TEXT_FORMAT_FILES
normalize-helpers/text/apache.access.format
normalize-helpers/text/bluecoat.format
normalize-helpers/text/bind.query.format
normalize-helpers/text/iis.format
normalize-helpers/text/iptables.format
normalize-helpers/text/ssh-login.format
normalize-helpers/text/squid.format
normalize-helpers/text/syslog.format
)

SET(PLUGINS_LIST
${NORMALIZE_FILTERS_LIB_LIST}
PARENT_SCOPE
)

INSTALL(FILES ${NORMALIZE_TEXT_FORMAT_FILES} DESTINATION normalize-helpers/text)
INSTALL(FILES ${NORMALIZE_PCAP_FORMAT_FILES} DESTINATION normalize-helpers/pcap)
INSTALL(FILES ${NORMALIZE_PERL_FORMAT_FILES} DESTINATION normalize-helpers/perl)
