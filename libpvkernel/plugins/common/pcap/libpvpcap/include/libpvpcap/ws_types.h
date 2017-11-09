/* *
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2017
 *
 */

#ifndef _ws_types_h
#define _ws_types_h

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

#include <unordered_map>

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
    {"FT_RELATIVE_TIME", "time"},
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

auto ws_map_type = [&](const std::string& type) -> std::string {
	const auto& it = ws_types_mapping.find(type);
	if (it != ws_types_mapping.end()) {
		return it->second;
	} else {
		// fallback type for unkown types
		return "string";
	}
};

#endif
