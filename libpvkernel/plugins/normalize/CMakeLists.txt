#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

#
# Input type plugins 
#
include_directories(AFTER ${PCAP_INCLUDE_DIRS})

set(NORMALIZE_FILTERS_DIST_FILES
)

macro(declare_library target)
	get_target_property(LIB_NAME ${target} LOCATION)
	set(NORMALIZE_FILTERS_DIST_LIST
		${NORMALIZE_FILTERS_DIST_LIST}
		${target})
	set(_NORMALIZE_FILTERS_LIB_LIST
		${_NORMALIZE_FILTERS_LIB_LIST}
		${LIB_NAME})
endmacro(declare_library)

# Splitters

#add_library(normalize_filter_splitter_regexp SHARED splitter_regexp/PVFieldSplitterRegexp.cpp splitter_regexp/PVFieldSplitterRegexp.h splitter_regexp/register.cpp)
#target_link_libraries(normalize_filter_splitter_regexp pvfilter pvcore)

SET(CSV_MOC_HEADERS
splitter_csv/PVFieldSplitterCSVParamWidget.h
)

QT4_WRAP_CPP(CSV_MOC_SRCS ${CSV_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
add_library(normalize_filter_splitter_csv SHARED splitter_csv/PVFieldSplitterCSV.cpp splitter_csv/PVFieldSplitterCSV.h splitter_csv/PVFieldSplitterCSVParamWidget.cpp ${CSV_MOC_SRCS} splitter_csv/libcsv.cpp splitter_csv/register.cpp)
target_link_libraries(normalize_filter_splitter_csv pvkernel ${QT_LIBRARIES})
declare_library(normalize_filter_splitter_csv)

SET(REGEXP_MOC_HEADERS
splitter_regexp/PVFieldSplitterRegexpParamWidget.h
)
QT4_WRAP_CPP(REGEXP_MOC_SRCS ${REGEXP_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
add_library(normalize_filter_splitter_regexp SHARED splitter_regexp/PVFieldSplitterRegexp.cpp splitter_regexp/PVFieldSplitterRegexp.h splitter_regexp/PVFieldSplitterRegexpParamWidget.cpp ${REGEXP_MOC_SRCS} splitter_regexp/register.cpp)
target_link_libraries(normalize_filter_splitter_regexp pvkernel ${QT_LIBRARIES} ${Boost_LIBRARIES} ${ICU_LIBRARY})
declare_library(normalize_filter_splitter_regexp)

add_definitions(-DFURL_UTF16_CHAR)
add_library(normalize_filter_splitter_url SHARED splitter_url/PVFieldSplitterURL.cpp splitter_url/PVFieldSplitterURL.h splitter_url/register.cpp)
target_link_libraries(normalize_filter_splitter_url pvkernel ${QT_LIBRARIES} /usr/local/lib/libfurll.a)
declare_library(normalize_filter_splitter_url)

add_library(normalize_filter_splitter_pcap SHARED splitter_pcap/PVFieldSplitterPcapPacket.cpp splitter_pcap/PVFieldSplitterPcapPacket.h splitter_pcap/register.cpp)
target_link_libraries(normalize_filter_splitter_pcap pvkernel ${QT_LIBRARIES} ${PCAP_LIBRARIES} ${WINSOCK_LIB})
declare_library(normalize_filter_splitter_pcap)

add_library(normalize_filter_splitter_ironport_mail SHARED splitter_ironport_mail/PVFieldSplitterIronPortMail.cpp splitter_ironport_mail/PVFieldSplitterIronPortMailParamWidget.cpp splitter_ironport_mail/register.cpp)
target_link_libraries(normalize_filter_splitter_ironport_mail pvkernel ${QT_LIBRARIES})
declare_library(normalize_filter_splitter_ironport_mail)

SET(GUIDTOIP_MOC_HEADERS
converter_GUID_to_IP/PVFieldConverterGUIDToIPParamWidget.h
)
QT4_WRAP_CPP(GUIDTOIP_MOC_SRCS ${GUIDTOIP_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
add_library(normalize_filter_converter_GUID_to_IP SHARED converter_GUID_to_IP/PVFieldGUIDToIP.cpp converter_GUID_to_IP/PVFieldConverterGUIDToIPParamWidget.cpp ${GUIDTOIP_MOC_SRCS} converter_GUID_to_IP/register.cpp)
target_link_libraries(normalize_filter_converter_GUID_to_IP pvkernel ${QT_LIBRARIES})
declare_library(normalize_filter_converter_GUID_to_IP)

SET(VALUEMAPPER_MOC_HEADERS
converter_value_mapper/PVFieldConverterValueMapperParamWidget.h
)
QT4_WRAP_CPP(VALUEMAPPER_MOC_SRCS ${VALUEMAPPER_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
add_library(normalize_filter_converter_value_mapper SHARED converter_value_mapper/PVFieldConverterValueMapper.cpp converter_value_mapper/PVFieldConverterValueMapperParamWidget.cpp converter_value_mapper/libcsv.cpp ${VALUEMAPPER_MOC_SRCS} converter_value_mapper/register.cpp)
target_link_libraries(normalize_filter_converter_value_mapper pvkernel ${QT_LIBRARIES})
declare_library(normalize_filter_converter_value_mapper)

SET(SPLITTERIP_MOC_HEADERS
splitter_ip/PVFieldSplitterIPParamWidget.h
)
QT4_WRAP_CPP(SPLITTERIP_MOC_SRCS ${SPLITTERIP_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
add_library(normalize_filter_splitter_ip SHARED splitter_ip/PVFieldSplitterIP.cpp splitter_ip/PVFieldSplitterIPParamWidget.cpp ${SPLITTERIP_MOC_SRCS} splitter_ip/register.cpp)
target_link_libraries(normalize_filter_splitter_ip pvkernel ${QT_LIBRARIES})
declare_library(normalize_filter_splitter_ip)

# "grep"

add_library(normalize_filter_grep_regexp SHARED grep_regexp/PVFieldFilterRegexpGrep.cpp grep_regexp/PVFieldFilterRegexpGrep.h grep_regexp/register.cpp)
target_link_libraries(normalize_filter_grep_regexp pvkernel ${QT_LIBRARIES})
declare_library(normalize_filter_grep_regexp)

SET(DUPLICATE_MOC_HEADERS
splitter_duplicate/PVFieldSplitterDuplicateParamWidget.h
)

QT4_WRAP_CPP(DUPLICATE_MOC_SRCS ${DUPLICATE_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
add_library(normalize_filter_splitter_duplicate SHARED splitter_duplicate/PVFieldDuplicate.cpp splitter_duplicate/PVFieldDuplicate.h splitter_duplicate/PVFieldSplitterDuplicateParamWidget.cpp ${DUPLICATE_MOC_SRCS} splitter_duplicate/register.cpp)
target_link_libraries(normalize_filter_splitter_duplicate pvkernel ${QT_LIBRARIES})
declare_library(normalize_filter_splitter_duplicate)

SET(MACADDRESS_MOC_HEADERS
splitter_mac_address/PVFieldSplitterMacAddressParamWidget.h
)

QT4_WRAP_CPP(MACADDRESS_MOC_SRCS ${MACADDRESS_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
add_library(normalize_filter_splitter_mac_address SHARED splitter_mac_address/PVFieldSplitterMacAddress.cpp splitter_mac_address/PVFieldSplitterMacAddress.h splitter_mac_address/PVFieldSplitterMacAddressParamWidget.cpp ${MACADDRESS_MOC_SRCS} splitter_mac_address/register.cpp)
target_link_libraries(normalize_filter_splitter_mac_address pvkernel ${QT_LIBRARIES})
declare_library(normalize_filter_splitter_mac_address)

SET(DNSFQDN_MOC_HEADERS
splitter_dns_fqdn/PVFieldSplitterDnsFqdnParamWidget.h
)

QT4_WRAP_CPP(DNSFQDN_MOC_SRCS ${DNSFQDN_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
add_library(normalize_filter_splitter_dns_fqdn SHARED splitter_dns_fqdn/PVFieldSplitterDnsFqdn.cpp splitter_dns_fqdn/PVFieldSplitterDnsFqdn.h splitter_dns_fqdn/PVFieldSplitterDnsFqdnParamWidget.cpp ${DNSFQDN_MOC_SRCS} splitter_dns_fqdn/register.cpp)
target_link_libraries(normalize_filter_splitter_dns_fqdn pvkernel ${QT_LIBRARIES})
declare_library(normalize_filter_splitter_dns_fqdn)

INSTALL(TARGETS ${NORMALIZE_FILTERS_DIST_LIST} LIBRARY DESTINATION normalize-filters
                                               RUNTIME DESTINATION normalize-filters)
INSTALL(FILES ${NORMALIZE_FILTERS_DIST_FILES} DESTINATION normalize-filters)

SET(NORMALIZE_FILTERS_LIB_LIST ${_NORMALIZE_FILTERS_LIB_LIST} PARENT_SCOPE)
