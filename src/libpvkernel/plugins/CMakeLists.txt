#
# @file
#
include_directories(BEFORE common)

add_subdirectory(normalize-filters)
add_subdirectory(input-types)
add_subdirectory(sources)

###############################################################################
# Ressource declaration
# These are default format files provided to users
###############################################################################

# install profile files
install(DIRECTORY common/pcap/profiles DESTINATION ${PVKERNEL_PLUGIN}/input-types/pcap)
