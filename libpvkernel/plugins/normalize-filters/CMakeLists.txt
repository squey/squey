#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

###############################################################################
# Normalize plugins 
###############################################################################

set(NORMALIZE_FILTERS_DIST_FILES
)

# Macro to list all plugins define for normalize-filter
macro(declare_library target)
	get_target_property(LIB_NAME ${target} LOCATION)
	set(NORMALIZE_FILTERS_DIST_LIST
		${NORMALIZE_FILTERS_DIST_LIST}
		${target})
endmacro(declare_library)

# Converter

################################################################################
## GUID_to_IP Converter
################################################################################
#
## Define ressources
#SET(GUIDTOIP_MOC_HEADERS
#converter_GUID_to_IP/PVFieldConverterGUIDToIPParamWidget.h
#)
#
#set(GUIDTOIP_SRC
#converter_GUID_to_IP/PVFieldGUIDToIP.cpp
#converter_GUID_to_IP/PVFieldConverterGUIDToIPParamWidget.cpp
#converter_GUID_to_IP/register.cpp
#)
#
## Generate Qt files
#QT5_WRAP_CPP(GUIDTOIP_MOC_SRCS ${GUIDTOIP_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
#
## Create lib
#add_library(normalize_filter_converter_GUID_to_IP SHARED ${GUIDTOIP_SRC} ${GUIDTOIP_MOC_SRCS})
#target_link_libraries(normalize_filter_converter_GUID_to_IP pvkernel)
#declare_library(normalize_filter_converter_GUID_to_IP)
#
################################################################################
## substitution Converter
################################################################################
#
## Define ressources
#SET(Substitution_MOC_HEADERS
#converter_substitution/PVFieldConverterSubstitutionParamWidget.h
#)
#
#SET(SUBSTITUTION_SRCS
#converter_substitution/PVFieldConverterSubstitution.cpp
#converter_substitution/PVFieldConverterSubstitutionParamWidget.cpp
#converter_substitution/libcsv.cpp
#converter_substitution/register.cpp
#)
#
## Generate Qt files
#QT5_WRAP_CPP(Substitution_MOC_SRCS ${Substitution_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
#
## Create lib
#add_library(normalize_filter_converter_substitution SHARED ${SUBSTITUTION_SRCS} ${Substitution_MOC_SRCS})
#target_link_libraries(normalize_filter_converter_substitution pvkernel)
#declare_library(normalize_filter_converter_substitution)
#
## Splitter

###############################################################################
# CSV Splitter
###############################################################################

# Declare ressources
SET(CSV_MOC_HEADERS
splitter_csv/PVFieldSplitterCSVParamWidget.h
)

set(CSV_SRC
splitter_csv/PVFieldSplitterCSV.cpp
splitter_csv/PVFieldSplitterCSVParamWidget.cpp
splitter_csv/register.cpp
	)

# Generate Qt files
QT5_WRAP_CPP(CSV_MOC_SRCS ${CSV_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

# Create lib
add_library(normalize_filter_splitter_csv SHARED ${CSV_SRC} ${CSV_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_csv pvkernel)
declare_library(normalize_filter_splitter_csv)

###############################################################################
# Regex Splitter
###############################################################################

# Declare ressources
SET(REGEXP_MOC_HEADERS
splitter_regexp/PVFieldSplitterRegexpParamWidget.h
)

set(REGEX_SRC
splitter_regexp/PVFieldSplitterRegexp.cpp
splitter_regexp/PVFieldSplitterRegexpParamWidget.cpp
splitter_regexp/register.cpp
)

# Genrate Qt files
QT5_WRAP_CPP(REGEXP_MOC_SRCS ${REGEXP_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

# Create lib
add_library(normalize_filter_splitter_regexp SHARED ${REGEX_SRC} ${REGEXP_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_regexp pvkernel ${Boost_LIBRARIES} ${ICU_LIBRARY})
declare_library(normalize_filter_splitter_regexp)

###############################################################################
# URL Splitter
###############################################################################

# Define ressources
set(URL_SRC
splitter_url/PVFieldSplitterURL.cpp
splitter_url/register.cpp
)

# Create lib
add_library(normalize_filter_splitter_url SHARED ${URL_SRC})
target_link_libraries(normalize_filter_splitter_url pvkernel ${FURL_STATIC_LIB})
declare_library(normalize_filter_splitter_url)

###############################################################################
# PCAP Splitter
###############################################################################

# Define ressources
SET(PCAP_SRC
splitter_pcap/PVFieldSplitterPcapPacket.cpp
splitter_pcap/register.cpp
)

include_directories(AFTER ${PCAP_INCLUDE_DIRS})

# Create lib
add_library(normalize_filter_splitter_pcap SHARED ${PCAP_SRC})
target_link_libraries(normalize_filter_splitter_pcap pvkernel ${PCAP_LIBRARIES} ${WINSOCK_LIB})
declare_library(normalize_filter_splitter_pcap)

################################################################################
## IRONPORT Splitter
################################################################################
#
## Define ressources
#set(IRONPORT_SRC
#splitter_ironport_mail/PVFieldSplitterIronPortMail.cpp
#splitter_ironport_mail/PVFieldSplitterIronPortMailParamWidget.cpp
#splitter_ironport_mail/register.cpp
#)
#
## Create lib
#add_library(normalize_filter_splitter_ironport_mail SHARED ${IRONPORT_SRC})
#target_link_libraries(normalize_filter_splitter_ironport_mail pvkernel)
#declare_library(normalize_filter_splitter_ironport_mail)
#
################################################################################
## key_value Splitter
################################################################################
#
## Define ressources
#SET(KEYVALUE_MOC_HEADERS
#splitter_key_value/PVFieldSplitterKeyValueParamWidget.h
#)
#
#SET(KEYVALUE_SRCS
#splitter_key_value/PVFieldSplitterKeyValue.cpp
#splitter_key_value/PVFieldSplitterKeyValueParamWidget.cpp
#splitter_key_value/register.cpp
#)
#
## Generate Qt files
#QT5_WRAP_CPP(KEYVALUE_MOC_SRCS ${KEYVALUE_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
#
## Create lib
#add_library(normalize_filter_splitter_key_value SHARED ${KEYVALUE_SRCS} ${KEYVALUE_MOC_SRCS})
#target_link_libraries(normalize_filter_splitter_key_value pvkernel)
#declare_library(normalize_filter_splitter_key_value)
#
###############################################################################
# IP Splitter
###############################################################################

# Define ressources
SET(SPLITTERIP_MOC_HEADERS
splitter_ip/PVFieldSplitterIPParamWidget.h
)

SET(IP_SRC
splitter_ip/PVFieldSplitterIP.cpp
splitter_ip/PVFieldSplitterIPParamWidget.cpp
splitter_ip/register.cpp
)

# Generate Qt files
QT5_WRAP_CPP(SPLITTERIP_MOC_SRCS ${SPLITTERIP_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

# Create lib
add_library(normalize_filter_splitter_ip SHARED ${IP_SRC} ${SPLITTERIP_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_ip pvkernel)
declare_library(normalize_filter_splitter_ip)

###############################################################################
# Duplicate Splitter
###############################################################################

# Define ressource
SET(DUPLICATE_MOC_HEADERS
splitter_duplicate/PVFieldSplitterDuplicateParamWidget.h
)

SET(DUPLICATE_SRC
splitter_duplicate/PVFieldDuplicate.cpp
splitter_duplicate/PVFieldSplitterDuplicateParamWidget.cpp
splitter_duplicate/register.cpp
)

# Generate Qt files
QT5_WRAP_CPP(DUPLICATE_MOC_SRCS ${DUPLICATE_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

# Create lib
add_library(normalize_filter_splitter_duplicate SHARED ${DUPLICATE_SRC} ${DUPLICATE_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_duplicate pvkernel)
declare_library(normalize_filter_splitter_duplicate)

################################################################################
## MacAdresse Splitter
################################################################################
#
## Define ressources
#SET(MACADDRESS_MOC_HEADERS
#splitter_mac_address/PVFieldSplitterMacAddressParamWidget.h
#)
#
#SET(MACADDRESS_SRC
#splitter_mac_address/PVFieldSplitterMacAddress.cpp
#splitter_mac_address/PVFieldSplitterMacAddressParamWidget.cpp
#splitter_mac_address/register.cpp
#)
#
##Generate Qt files
#QT5_WRAP_CPP(MACADDRESS_MOC_SRCS ${MACADDRESS_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
#
## Create lib
#add_library(normalize_filter_splitter_mac_address SHARED ${MACADDRESS_SRC} ${MACADDRESS_MOC_SRCS})
#target_link_libraries(normalize_filter_splitter_mac_address pvkernel)
#declare_library(normalize_filter_splitter_mac_address)
#
################################################################################
## DNSFQDN Splitter
################################################################################
#
## Define ressources
#SET(DNSFQDN_MOC_HEADERS
#splitter_dns_fqdn/PVFieldSplitterDnsFqdnParamWidget.h
#)
#
#SET(DNSFQDN_SRC
#splitter_dns_fqdn/PVFieldSplitterDnsFqdn.cpp
#splitter_dns_fqdn/PVFieldSplitterDnsFqdnParamWidget.cpp
#splitter_dns_fqdn/register.cpp
#)
#
## Generate Qt files
#QT5_WRAP_CPP(DNSFQDN_MOC_SRCS ${DNSFQDN_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
#
## Create lib
#add_library(normalize_filter_splitter_dns_fqdn SHARED ${DNSFQDN_SRC} ${DNSFQDN_MOC_SRCS})
#target_link_libraries(normalize_filter_splitter_dns_fqdn pvkernel)
#declare_library(normalize_filter_splitter_dns_fqdn)
#
################################################################################
## grep
################################################################################
#
## Define ressources
#SET(GREP_SRC
#grep_regexp/PVFieldFilterRegexpGrep.cpp
#grep_regexp/register.cpp
#)
#
## Create lib
#add_library(normalize_filter_grep_regexp SHARED ${GREP_SRC})
#target_link_libraries(normalize_filter_grep_regexp pvkernel)
#declare_library(normalize_filter_grep_regexp)

###############################################################################
# Install normalize plugins
###############################################################################

INSTALL(TARGETS ${NORMALIZE_FILTERS_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/normalize-filters
					       RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/normalize-filters)
INSTALL(FILES ${NORMALIZE_FILTERS_DIST_FILES} DESTINATION ${SHARE_RESSOURCES}/normalize-filters)
