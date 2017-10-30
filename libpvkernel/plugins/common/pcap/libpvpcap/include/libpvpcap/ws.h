/*!
 * \file
 * \brief Interface to manage the thsark commands.
 *
 * All the thsark commands are dealt here.
 *
 * \copyright (C) Philippe Saade - Justin Teliet 2014
 * \copyright (C) ESI Group INENDI 2017
 */

#ifndef WS_H
#define WS_H

#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "rapidjson/document.h"

#include "shell.h"

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

/*
Extracted from tshark -G ftypes

FT_NONE Label
FT_PROTOCOL     Protocol
FT_BOOLEAN      Boolean
FT_UINT8        Unsigned integer, 1 byte
FT_UINT16       Unsigned integer, 2 bytes
FT_UINT24       Unsigned integer, 3 bytes
FT_UINT32       Unsigned integer, 4 bytes
FT_UINT64       Unsigned integer, 8 bytes
FT_INT8 Signed integer, 1 byte
FT_INT16        Signed integer, 2 bytes
FT_INT24        Signed integer, 3 bytes
FT_INT32        Signed integer, 4 bytes
FT_INT64        Signed integer, 8 bytes
FT_FLOAT        Floating point (single-precision)
FT_DOUBLE       Floating point (double-precision)
FT_ABSOLUTE_TIME        Date and time
FT_RELATIVE_TIME        Time offset
FT_STRING       Character string
FT_STRINGZ      Character string
FT_UINT_STRING  Character string
FT_ETHER        Ethernet or other MAC address
FT_BYTES        Sequence of bytes
FT_UINT_BYTES   Sequence of bytes
FT_IPv4 IPv4 address
FT_IPv6 IPv6 address
FT_IPXNET       IPX network number
FT_FRAMENUM     Frame number
FT_PCRE Compiled Perl-Compatible Regular Expression (GRegex) object
FT_GUID Globally Unique Identifier
FT_OID  ASN.1 object identifier
FT_EUI64        EUI64 address
FT_AX25 AX.25 address
FT_VINES        VINES address
FT_REL_OID      ASN.1 relative object identifier
FT_SYSTEM_ID    OSI System-ID
FT_STRINGZPAD   Character string

Other explanation on https://www.wireshark.org/docs/wsar_html/epan/ftypes_8h_source.html
*/

// type mapping between wireshark and inspector
static const std::unordered_map<std::string, std::string> ws_types_mapping = {
    {"FT_NONE Label", "string"},
    {"FT_PROTOCOL", "string"},
    {"FT_BOOLEAN", "string"}, /* not supported */
    {"FT_UINT8", "number_uint8"},
    {"FT_UINT16", "number_uint16"},
    {"FT_UINT24", "number_uint32"},
    {"FT_UINT32", "number_uint32"},
    {"FT_UINT64", "number_uint64"},
    {"FT_INT8", "number_int8"},
    {"FT_INT16", "number_int16"},
    {"FT_INT24", "number_int32"},
    {"FT_INT32", "number_int32"},
    {"FT_INT64", "number_int64"},
    {"FT_FLOAT", "number_float"},
    {"FT_DOUBLE", "number_double"},
    {"FT_ABSOLUTE_TIME", "time"},
    {"FT_RELATIVE_TIME", "duration"},
    {"FT_STRING", "string"},
    {"FT_STRINGZ", "string"},
    {"FT_UINT_STRING", "string"},
    {"FT_ETHER", "mac_address"},
    {"FT_BYTES", "string"},      /* not supported - number_int8 ? */
    {"FT_UINT_BYTES", "string"}, /* not supported - number_uint8 ? */
    {"FT_IPv4", "ipv4"},
    {"FT_IPv6", "ipv6"},
    {"FT_IPXNET", "number_uint32"},
    {"FT_FRAMENUM", "number_uint32"},
    {"FT_PCRE", "string"},      /* not supported */
    {"FT_GUID", "string"},      /* not supported */
    {"FT_OID", "string"},       /* not supported */
    {"FT_EUI64", "string"},     /* not supported */
    {"FT_AX25", "string"},      /* not supported */
    {"FT_VINES", "string"},     /* not supported */
    {"FT_REL_OID", "string"},   /* not supported */
    {"FT_SYSTEM_ID", "string"}, /* not supported */
    {"FT_STRINGZPAD", "string"},
};

static const std::unordered_set<std::string> ws_disabled_fields = {
    "frame.offset_shift",
    "frame.time_delta",
    "frame.time_delta_displayed",
    "frame.ref_time",
    "frame.number",
    "frame.marked",
    "frame.ignored",
    "frame.coloring_rule.name",
    "frame.coloring_rule.string",
    "http.response.line",
    "http.request.line",
    "tcp.stream",
    "udp.stream",
};

inline std::string ws_map_type(const std::string& type)
{
	const auto& it = ws_types_mapping.find(type);
	if (it != ws_types_mapping.end()) {
		return it->second;
	} else {
		// fallback type for unkown types
		return "string";
	}
};

// transform to inspector format quote
static const std::unordered_map<std::string, std::string> ws_map_quote = {
    {"d", "\""}, {"s", "'"}, {"n", ""},
};

// transform to inspector format quote
static const std::unordered_map<std::string, std::string> ws_map_special_fields = {
    {"source", "_ws.col.Source"},
    {"destination", "_ws.col.Destination"},
    {"protocol", "_ws.col.Protocol"},
    {"info", "_ws.col.Info"},
};

std::string ws_protocols_dict_path();

/**
 * Get the list of wireshark's known protocols' fields.
 *
 * @return list of fields.
 */
std::vector<std::string> ws_get_tshark_fields();

/**
 * Get the list of wireshark's known protocols.
 *
 * @return list of protocols.
 */
std::vector<std::string> ws_get_tshark_protocols();

/**
 * Build a dictionnary of all known protocols and fields from tshark and store
 * it in a json format file.
 *
 * @param protocols_dict_file file to save json string.
 */
void ws_create_protocols_dict(std::string const& protocols_dict_file);

rapidjson::Document ws_parse_protocol_dict(const std::string& protocols_dict_file);

/**
 * Build json tree structure from a pcap file.
 *
 * @param source pcap file name.
 * @param  document rapidjson document.
 * @param  protocols_dict_file a json file of all known protocols and fields from tshark
 */
rapidjson::Document ws_create_protocols_tree(const std::string& source,
                                             const rapidjson::Document& protocols_dict);

void ws_merge_protocols_tree(rapidjson::Value& dst,
                             const rapidjson::Value& src,
                             rapidjson::Document::AllocatorType& allocator);

void ws_enrich_protocols_tree(rapidjson::Document& enriched_protocols_tree,
                              const rapidjson::Document& protocols_tree,
                              const rapidjson::Document& protocols_dict);

/**
 *
 */
rapidjson::Document ws_get_selected_fields(const rapidjson::Document& json_data);

/**
 *
 */
std::vector<std::string> ws_get_cmdline_opts(rapidjson::Document& json_data);

} /* namespace pvpcap */

#endif // WS_H
