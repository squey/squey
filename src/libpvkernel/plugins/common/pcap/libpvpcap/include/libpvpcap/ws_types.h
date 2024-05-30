/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
