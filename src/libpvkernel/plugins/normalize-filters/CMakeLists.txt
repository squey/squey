#
# @file
#

# For QtCore5Compat
find_package(Qt6 COMPONENTS Core5Compat REQUIRED)

###############################################################################
# Normalize plugins
###############################################################################

set(NORMALIZE_FILTERS_DIST_FILES
)

# Macro to list all plugins define for normalize-filter
macro(declare_library target)
	set(NORMALIZE_FILTERS_DIST_LIST
		${NORMALIZE_FILTERS_DIST_LIST}
		${target})
endmacro()

# Converter

###############################################################################
# GUID_to_IP Converter
###############################################################################

set(GUIDTOIP_SRC
converter_GUID_to_IP/PVFieldGUIDToIP.cpp
converter_GUID_to_IP/PVFieldConverterGUIDToIPParamWidget.cpp
converter_GUID_to_IP/register.cpp
)

# Create lib
add_library(normalize_filter_converter_GUID_to_IP SHARED ${GUIDTOIP_SRC})
target_link_libraries(normalize_filter_converter_GUID_to_IP PRIVATE pvkernel)
declare_library(normalize_filter_converter_GUID_to_IP)

###############################################################################
# substitution Converter
###############################################################################

# Define ressources
set(Substitution_MOC_HEADERS
converter_substitution/PVFieldConverterSubstitutionParamWidget.h
)

set(SUBSTITUTION_SRCS
converter_substitution/PVFieldConverterSubstitution.cpp
converter_substitution/PVFieldConverterSubstitutionParamWidget.cpp
converter_substitution/register.cpp
)

# Generate Qt files
qt6_wrap_cpp(Substitution_MOC_SRCS ${Substitution_MOC_HEADERS} OPTIONS)

# Create lib
add_library(normalize_filter_converter_substitution SHARED ${SUBSTITUTION_SRCS} ${Substitution_MOC_SRCS})
target_link_libraries(normalize_filter_converter_substitution PRIVATE pvkernel)
declare_library(normalize_filter_converter_substitution)

# Splitter

###############################################################################
# struct Converter
###############################################################################

# Define ressources
set(Struct_MOC_HEADERS
converter_struct/PVFieldConverterStructParamWidget.h
)

set(STRUCT_SRCS
converter_struct/PVFieldConverterStruct.cpp
converter_struct/PVFieldConverterStructParamWidget.cpp
converter_struct/register.cpp
)

# Generate Qt files
qt6_wrap_cpp(Struct_MOC_SRCS ${Struct_MOC_HEADERS} OPTIONS)

# Create lib
add_library(normalize_filter_converter_struct SHARED ${STRUCT_SRCS} ${Struct_MOC_SRCS})
target_link_libraries(normalize_filter_converter_struct PRIVATE pvkernel)
declare_library(normalize_filter_converter_struct)

# Splitter

###############################################################################
# CSV Splitter
###############################################################################

# Declare ressources
set(CSV_MOC_HEADERS
splitter_csv/PVFieldSplitterCSVParamWidget.h
)

set(CSV_SRC
splitter_csv/PVFieldSplitterCSV.cpp
splitter_csv/PVFieldSplitterCSVParamWidget.cpp
splitter_csv/register.cpp
	)

# Generate Qt files
qt6_wrap_cpp(CSV_MOC_SRCS ${CSV_MOC_HEADERS} OPTIONS)

# Create lib
add_library(normalize_filter_splitter_csv SHARED ${CSV_SRC} ${CSV_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_csv PRIVATE pvkernel)
declare_library(normalize_filter_splitter_csv)

###############################################################################
# Regex Splitter
###############################################################################

# Declare ressources
set(REGEXP_MOC_HEADERS
splitter_regexp/PVFieldSplitterRegexpParamWidget.h
)

set(REGEX_SRC
splitter_regexp/PVFieldSplitterRegexp.cpp
splitter_regexp/PVFieldSplitterRegexpParamWidget.cpp
splitter_regexp/register.cpp
)

# Genrate Qt files
qt6_wrap_cpp(REGEXP_MOC_SRCS ${REGEXP_MOC_HEADERS} OPTIONS)

# Create lib
add_library(normalize_filter_splitter_regexp SHARED ${REGEX_SRC} ${REGEXP_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_regexp PRIVATE pvkernel ${Boost_LIBRARIES} ${ICU_LIBRARY})
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
target_link_libraries(normalize_filter_splitter_url PRIVATE pvkernel furls-utf8)
declare_library(normalize_filter_splitter_url)

###############################################################################
# key_value Splitter
###############################################################################

# Define ressources
set(KEYVALUE_MOC_HEADERS
splitter_key_value/PVFieldSplitterKeyValueParamWidget.h
)

set(KEYVALUE_SRCS
splitter_key_value/PVFieldSplitterKeyValue.cpp
splitter_key_value/PVFieldSplitterKeyValueParamWidget.cpp
splitter_key_value/register.cpp
)

# Generate Qt files
qt6_wrap_cpp(KEYVALUE_MOC_SRCS ${KEYVALUE_MOC_HEADERS} OPTIONS)

# Create lib
add_library(normalize_filter_splitter_key_value SHARED ${KEYVALUE_SRCS} ${KEYVALUE_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_key_value PRIVATE pvkernel)
declare_library(normalize_filter_splitter_key_value)

###############################################################################
# IP Splitter
###############################################################################

# Define ressources
set(SPLITTERIP_MOC_HEADERS
splitter_ip/PVFieldSplitterIPParamWidget.h
)

set(IP_SRC
splitter_ip/PVFieldSplitterIP.cpp
splitter_ip/PVFieldSplitterIPParamWidget.cpp
splitter_ip/register.cpp
)

# Generate Qt files
qt6_wrap_cpp(SPLITTERIP_MOC_SRCS ${SPLITTERIP_MOC_HEADERS} OPTIONS)

# Create lib
add_library(normalize_filter_splitter_ip SHARED ${IP_SRC} ${SPLITTERIP_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_ip PRIVATE pvkernel)
declare_library(normalize_filter_splitter_ip)

###############################################################################
# Duplicate Splitter
###############################################################################

# Define ressource
set(DUPLICATE_MOC_HEADERS
splitter_duplicate/PVFieldSplitterDuplicateParamWidget.h
)

set(DUPLICATE_SRC
splitter_duplicate/PVFieldDuplicate.cpp
splitter_duplicate/PVFieldSplitterDuplicateParamWidget.cpp
splitter_duplicate/register.cpp
)

# Generate Qt files
qt6_wrap_cpp(DUPLICATE_MOC_SRCS ${DUPLICATE_MOC_HEADERS} OPTIONS)

# Create lib
add_library(normalize_filter_splitter_duplicate SHARED ${DUPLICATE_SRC} ${DUPLICATE_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_duplicate PRIVATE pvkernel)
declare_library(normalize_filter_splitter_duplicate)

###############################################################################
# Mac Adresse Splitter
###############################################################################

# Define ressources
set(MACADDRESS_MOC_HEADERS
splitter_mac_address/PVFieldSplitterMacAddressParamWidget.h
)

set(MACADDRESS_SRC
splitter_mac_address/PVFieldSplitterMacAddress.cpp
splitter_mac_address/PVFieldSplitterMacAddressParamWidget.cpp
splitter_mac_address/register.cpp
)

#Generate Qt files
qt6_wrap_cpp(MACADDRESS_MOC_SRCS ${MACADDRESS_MOC_HEADERS} OPTIONS)

# Create lib
add_library(normalize_filter_splitter_mac_address SHARED ${MACADDRESS_SRC} ${MACADDRESS_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_mac_address PRIVATE pvkernel)
declare_library(normalize_filter_splitter_mac_address)

###############################################################################
# DNS FQDN Splitter
###############################################################################

# Define ressources
set(DNSFQDN_MOC_HEADERS
splitter_dns_fqdn/PVFieldSplitterDnsFqdnParamWidget.h
)

set(DNSFQDN_SRC
splitter_dns_fqdn/PVFieldSplitterDnsFqdn.cpp
splitter_dns_fqdn/PVFieldSplitterDnsFqdnParamWidget.cpp
splitter_dns_fqdn/register.cpp
)

# Generate Qt files
qt6_wrap_cpp(DNSFQDN_MOC_SRCS ${DNSFQDN_MOC_HEADERS} OPTIONS)

# Create lib
add_library(normalize_filter_splitter_dns_fqdn SHARED ${DNSFQDN_SRC} ${DNSFQDN_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_dns_fqdn PRIVATE pvkernel)
declare_library(normalize_filter_splitter_dns_fqdn)

###############################################################################
# DNS FQDN Splitter
###############################################################################

# Define ressources
set(LENGTH_MOC_HEADERS
splitter_length/PVFieldSplitterLengthParamWidget.h
)

set(LENGTH_SRC
splitter_length/PVFieldSplitterLength.cpp
splitter_length/PVFieldSplitterLengthParamWidget.cpp
splitter_length/register.cpp
)

# Generate Qt files
qt6_wrap_cpp(LENGTH_MOC_SRCS ${LENGTH_MOC_HEADERS} OPTIONS)

# Create lib
add_library(normalize_filter_splitter_length SHARED ${LENGTH_SRC} ${LENGTH_MOC_SRCS})
target_link_libraries(normalize_filter_splitter_length PRIVATE pvkernel)
declare_library(normalize_filter_splitter_length)

###############################################################################
# grep Filter
###############################################################################

# Define ressources
set(GREP_SRC
grep_regexp/PVFieldFilterRegexpGrep.cpp
grep_regexp/register.cpp
)

# Create lib
add_library(normalize_filter_grep_regexp SHARED ${GREP_SRC})
target_link_libraries(normalize_filter_grep_regexp PRIVATE pvkernel)
declare_library(normalize_filter_grep_regexp)

###############################################################################
# Install normalize plugins
###############################################################################

install(TARGETS ${NORMALIZE_FILTERS_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/normalize-filters
					       RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/normalize-filters)
install(FILES ${NORMALIZE_FILTERS_DIST_FILES} DESTINATION ${SHARE_RESSOURCES}/normalize-filters)
