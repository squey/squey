#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

add_subdirectory(normalize-filters)
add_subdirectory(input_types)
add_subdirectory(sources)

###############################################################################
# Ressource declaration
# These are default format files provided to users
###############################################################################

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

INSTALL(FILES ${NORMALIZE_TEXT_FORMAT_FILES} DESTINATION ${SHARE_RESSOURCES}/normalize-helpers/text)
