//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <libpvpcap/flow.h>

#include <algorithm>
#include <cstring>
#include <string_view>

namespace pvpcap
{

namespace
{

// Link types, as pcap and pcapng files number them.
constexpr uint32_t LINKTYPE_NULL = 0;
constexpr uint32_t LINKTYPE_ETHERNET = 1;
constexpr uint32_t LINKTYPE_RAW_BSD = 12;
constexpr uint32_t LINKTYPE_RAW_OPENBSD = 14;
constexpr uint32_t LINKTYPE_RAW = 101;
constexpr uint32_t LINKTYPE_LOOP = 108;
constexpr uint32_t LINKTYPE_LINUX_SLL = 113;
constexpr uint32_t LINKTYPE_IPV4 = 228;
constexpr uint32_t LINKTYPE_IPV6 = 229;
constexpr uint32_t LINKTYPE_LINUX_SLL2 = 276;

constexpr uint16_t ETHERTYPE_IPV4 = 0x0800;
constexpr uint16_t ETHERTYPE_IPV6 = 0x86dd;
constexpr uint16_t ETHERTYPE_VLAN = 0x8100;
constexpr uint16_t ETHERTYPE_QINQ = 0x88a8;
constexpr uint16_t ETHERTYPE_QINQ_OLD = 0x9100;
constexpr uint16_t ETHERTYPE_QINQ_OLD2 = 0x9200;
constexpr uint16_t ETHERTYPE_MPLS = 0x8847;
constexpr uint16_t ETHERTYPE_MPLS_MULTICAST = 0x8848;
constexpr uint16_t ETHERTYPE_PPPOE_SESSION = 0x8864;
constexpr uint16_t PPP_IPV4 = 0x0021;
constexpr uint16_t PPP_IPV6 = 0x0057;

constexpr uint8_t PROTOCOL_ICMP = 1;
constexpr uint8_t PROTOCOL_TCP = 6;
constexpr uint8_t PROTOCOL_UDP = 17;
constexpr uint8_t PROTOCOL_DCCP = 33;
constexpr uint8_t PROTOCOL_IPV6_ROUTING = 43;
constexpr uint8_t PROTOCOL_IPV6_FRAGMENT = 44;
constexpr uint8_t PROTOCOL_AH = 51;
constexpr uint8_t PROTOCOL_ICMPV6 = 58;
constexpr uint8_t PROTOCOL_IPV6_HOP_BY_HOP = 0;
constexpr uint8_t PROTOCOL_IPV6_DESTINATION = 60;
constexpr uint8_t PROTOCOL_IPV6_MOBILITY = 135;
constexpr uint8_t PROTOCOL_SCTP = 132;
constexpr uint8_t PROTOCOL_UDP_LITE = 136;
constexpr uint8_t PROTOCOL_HIP = 139;
constexpr uint8_t PROTOCOL_SHIM6 = 140;

//! Any more is not what a real packet carries, but what a crafted one loops on.
constexpr int MOST_NESTED_HEADERS = 16;

class bytes
{
  public:
	explicit bytes(std::span<const uint8_t> data) : _data(data) {}

	bool has(size_t offset, size_t count) const
	{
		return offset <= _data.size() and count <= _data.size() - offset;
	}
	uint8_t u8(size_t offset) const { return _data[offset]; }
	uint16_t u16(size_t offset) const
	{
		return static_cast<uint16_t>((_data[offset] << 8) | _data[offset + 1]);
	}
	uint32_t u32(size_t offset) const
	{
		return (uint32_t(_data[offset]) << 24) | (uint32_t(_data[offset + 1]) << 16) |
		       (uint32_t(_data[offset + 2]) << 8) | uint32_t(_data[offset + 3]);
	}
	const uint8_t* at(size_t offset) const { return _data.data() + offset; }

  private:
	std::span<const uint8_t> _data;
};

bool has_ports(uint8_t protocol)
{
	return protocol == PROTOCOL_TCP or protocol == PROTOCOL_UDP or protocol == PROTOCOL_SCTP or
	       protocol == PROTOCOL_DCCP or protocol == PROTOCOL_UDP_LITE;
}

//! Orders the endpoints, so that both directions of a conversation get the same key.
void order(flow_key& key)
{
	const int by_address = std::memcmp(key.address_a.data(), key.address_b.data(), 16);
	if (by_address > 0 or (by_address == 0 and key.port_a > key.port_b)) {
		std::swap(key.address_a, key.address_b);
		std::swap(key.port_a, key.port_b);
	}
}

flow_key not_ip(uint16_t link)
{
	flow_key key;
	key.link = link;
	return key;
}

struct ip_header {
	flow_key key;       //!< endpoints and protocol, no ports yet
	size_t payload = 0; //!< offset of what follows the IP headers
	bool fragment = false;
	bool first_fragment = false;
	uint32_t datagram_id = 0;
	bool valid = false;
};

ip_header ipv4_header(const bytes& b, size_t offset)
{
	ip_header ip;
	if (not b.has(offset, 20) or (b.u8(offset) >> 4) != 4) {
		return ip;
	}
	const size_t header_length = size_t(b.u8(offset) & 0x0f) * 4;
	if (header_length < 20) {
		return ip;
	}
	ip.key.family = 4;
	ip.key.protocol = b.u8(offset + 9);
	std::memcpy(ip.key.address_a.data(), b.at(offset + 12), 4);
	std::memcpy(ip.key.address_b.data(), b.at(offset + 16), 4);
	const uint16_t fragmentation = b.u16(offset + 6);
	const bool more_fragments = fragmentation & 0x2000;
	const uint16_t fragment_offset = fragmentation & 0x1fff;
	ip.fragment = more_fragments or fragment_offset != 0;
	ip.first_fragment = ip.fragment and fragment_offset == 0;
	ip.datagram_id = b.u16(offset + 4);
	ip.payload = offset + header_length;
	ip.valid = true;
	return ip;
}

ip_header ipv6_header(const bytes& b, size_t offset)
{
	ip_header ip;
	if (not b.has(offset, 40) or (b.u8(offset) >> 4) != 6) {
		return ip;
	}
	ip.key.family = 6;
	std::memcpy(ip.key.address_a.data(), b.at(offset + 8), 16);
	std::memcpy(ip.key.address_b.data(), b.at(offset + 24), 16);
	uint8_t next = b.u8(offset + 6);
	size_t at = offset + 40;
	for (int i = 0; i < MOST_NESTED_HEADERS; i++) {
		if (next == PROTOCOL_IPV6_HOP_BY_HOP or next == PROTOCOL_IPV6_ROUTING or
		    next == PROTOCOL_IPV6_DESTINATION or next == PROTOCOL_IPV6_MOBILITY or
		    next == PROTOCOL_HIP or next == PROTOCOL_SHIM6) {
			if (not b.has(at, 2)) {
				break;
			}
			next = b.u8(at);
			at += (size_t(b.u8(at + 1)) + 1) * 8;
		} else if (next == PROTOCOL_AH) {
			if (not b.has(at, 2)) {
				break;
			}
			next = b.u8(at);
			at += (size_t(b.u8(at + 1)) + 2) * 4;
		} else if (next == PROTOCOL_IPV6_FRAGMENT) {
			if (not b.has(at, 8)) {
				break;
			}
			const uint16_t fragmentation = b.u16(at + 2);
			const bool more_fragments = fragmentation & 0x0001;
			const uint16_t fragment_offset = fragmentation >> 3;
			ip.fragment = more_fragments or fragment_offset != 0;
			ip.first_fragment = ip.fragment and fragment_offset == 0;
			ip.datagram_id = b.u32(at + 4);
			next = b.u8(at);
			at += 8;
		} else {
			break;
		}
	}
	ip.key.protocol = next;
	ip.payload = at;
	ip.valid = true;
	return ip;
}

ip_header ip_header_at(const bytes& b, size_t offset)
{
	if (not b.has(offset, 1)) {
		return {};
	}
	return (b.u8(offset) >> 4) == 4 ? ipv4_header(b, offset) : ipv6_header(b, offset);
}

void read_ports(const bytes& b, const ip_header& ip, flow_key& key)
{
	if (has_ports(ip.key.protocol) and b.has(ip.payload, 4)) {
		key.port_a = b.u16(ip.payload);
		key.port_b = b.u16(ip.payload + 2);
	}
}

bool is_icmp_error(const ip_header& ip, const bytes& b)
{
	if (not b.has(ip.payload, 1)) {
		return false;
	}
	const uint8_t type = b.u8(ip.payload);
	if (ip.key.protocol == PROTOCOL_ICMP) {
		return type == 3 or type == 4 or type == 5 or type == 11 or type == 12;
	}
	if (ip.key.protocol == PROTOCOL_ICMPV6) {
		return type >= 1 and type <= 4;
	}
	return false;
}

packet_flow ip_flow(const bytes& b, size_t offset)
{
	packet_flow flow;
	const ip_header ip = ip_header_at(b, offset);
	if (not ip.valid) {
		flow.key = not_ip(0);
		return flow;
	}

	flow.key = ip.key;
	if (ip.fragment) {
		flow.fragment = true;
		flow.first_fragment = ip.first_fragment;
		flow.datagram = ip.key;
		order(flow.datagram);
		flow.datagram_id = ip.datagram_id;
		if (ip.first_fragment) {
			read_ports(b, ip, flow.key);
		}
	} else if (is_icmp_error(ip, b)) {
		// The error quotes the header of the packet it is about, and the start of its
		// payload: that packet's conversation is where the error makes sense.
		const ip_header quoted = ip_header_at(b, ip.payload + 8);
		if (quoted.valid and quoted.key.family == ip.key.family) {
			flow.key = quoted.key;
			if (not quoted.fragment or quoted.first_fragment) {
				read_ports(b, quoted, flow.key);
			}
		}
	} else {
		read_ports(b, ip, flow.key);
	}
	order(flow.key);
	return flow;
}

/**
 * Skips the tags and the encapsulations that come before the network layer.
 *
 * @return the flow of what follows `offset`, whose ether type is `type`
 */
packet_flow ether_flow(const bytes& b, uint16_t type, size_t offset)
{
	for (int i = 0; i < MOST_NESTED_HEADERS; i++) {
		if (type == ETHERTYPE_VLAN or type == ETHERTYPE_QINQ or type == ETHERTYPE_QINQ_OLD or
		    type == ETHERTYPE_QINQ_OLD2) {
			if (not b.has(offset, 4)) {
				break;
			}
			type = b.u16(offset + 2);
			offset += 4;
		} else if (type == ETHERTYPE_PPPOE_SESSION) {
			if (not b.has(offset, 8)) {
				break;
			}
			const uint16_t ppp = b.u16(offset + 6);
			if (ppp != PPP_IPV4 and ppp != PPP_IPV6) {
				break;
			}
			type = ppp == PPP_IPV4 ? ETHERTYPE_IPV4 : ETHERTYPE_IPV6;
			offset += 8;
		} else if (type == ETHERTYPE_MPLS or type == ETHERTYPE_MPLS_MULTICAST) {
			// Down to the bottom of the label stack, past which only the version of an IP
			// header tells what is carried.
			bool bottom = false;
			while (not bottom and b.has(offset, 4)) {
				bottom = b.u8(offset + 2) & 0x01;
				offset += 4;
			}
			if (not bottom or not b.has(offset, 1)) {
				break;
			}
			const uint8_t version = b.u8(offset) >> 4;
			if (version != 4 and version != 6) {
				break;
			}
			type = version == 4 ? ETHERTYPE_IPV4 : ETHERTYPE_IPV6;
		} else {
			break;
		}
	}

	if (type == ETHERTYPE_IPV4 or type == ETHERTYPE_IPV6) {
		return ip_flow(b, offset);
	}
	packet_flow flow;
	flow.key = not_ip(type);
	return flow;
}

} // namespace

size_t flow_key_hash::operator()(const flow_key& key) const noexcept
{
	// The members leave no padding: hashing their bytes hashes nothing but their values.
	static_assert(sizeof(flow_key) == 40);
	return std::hash<std::string_view>()(
	    std::string_view(reinterpret_cast<const char*>(&key), sizeof(key)));
}

packet_flow flow_of(uint32_t link_type, std::span<const uint8_t> frame)
{
	const bytes b(frame);

	switch (link_type) {
	case LINKTYPE_ETHERNET:
		if (b.has(0, 14)) {
			return ether_flow(b, b.u16(12), 14);
		}
		break;
	case LINKTYPE_LINUX_SLL:
		if (b.has(0, 16)) {
			return ether_flow(b, b.u16(14), 16);
		}
		break;
	case LINKTYPE_LINUX_SLL2:
		if (b.has(0, 20)) {
			return ether_flow(b, b.u16(0), 20);
		}
		break;
	case LINKTYPE_NULL:
	case LINKTYPE_LOOP:
		// The address family comes in the byte order of whoever captured, and its value for
		// IPv6 differs between systems: the version of the IP header is what can be relied on.
		if (b.has(0, 5)) {
			return ip_flow(b, 4);
		}
		break;
	case LINKTYPE_RAW:
	case LINKTYPE_RAW_BSD:
	case LINKTYPE_RAW_OPENBSD:
	case LINKTYPE_IPV4:
	case LINKTYPE_IPV6:
		return ip_flow(b, 0);
	default:
		break;
	}

	// Kept together, the frames of a link this does not read are read as they would be in
	// the whole capture.
	packet_flow flow;
	flow.key = not_ip(static_cast<uint16_t>(link_type));
	flow.key.family = 0xff;
	return flow;
}

} // namespace pvpcap
