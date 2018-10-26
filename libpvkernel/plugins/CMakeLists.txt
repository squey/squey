#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015
include_directories(BEFORE common)
include_directories(${PVLICENSE_INCLUDE_DIR})

add_subdirectory(normalize-filters)
add_subdirectory(input-types)
add_subdirectory(sources)

###############################################################################
# Ressource declaration
# These are default format files provided to users
###############################################################################

# install profile files
INSTALL(DIRECTORY common/pcap/profiles DESTINATION ${PVKERNEL_PLUGIN}/input-types/pcap)
