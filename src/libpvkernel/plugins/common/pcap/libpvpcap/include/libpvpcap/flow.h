/* * MIT License
 *
 * © Squey, 2026
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

#ifndef LIBPVPCAP_FLOW_H
#define LIBPVPCAP_FLOW_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace pvpcap
{

/**
 * Identifies a conversation, the same way for both of its directions: the endpoints are
 * ordered, whichever sent the packet.
 *
 * The conversations of TCP, UDP, SCTP and DCCP are those of their ports. The other IP
 * protocols have one per pair of hosts, and what is not IP one per protocol of the link.
 */
struct flow_key {
	uint8_t family = 0;   //!< 4 or 6 for IP, 0 for what is not
	uint8_t protocol = 0; //!< IP protocol
	uint16_t link = 0;    //!< ether type, or link type, of what is not IP
	uint16_t port_a = 0;
	uint16_t port_b = 0;
	std::array<uint8_t, 16> address_a{}; //!< an IPv4 address takes the first 4 bytes
	std::array<uint8_t, 16> address_b{};

	bool operator==(const flow_key&) const = default;
};

struct flow_key_hash {
	size_t operator()(const flow_key& key) const noexcept;
};

/**
 * What a packet tells of the conversation it belongs to.
 */
struct packet_flow {
	flow_key key;

	/**
	 * The fragments of an IP datagram past the first hold no ports: they are only known by
	 * their datagram, which `key` names then, along with `datagram_id`.
	 */
	bool fragment = false;
	bool first_fragment = false; //!< the fragment that holds the ports, and gives them `key`
	flow_key datagram;           //!< of any fragment, without the ports
	uint32_t datagram_id = 0;
};

/**
 * The conversation a frame of the given link type belongs to.
 *
 * Reads no further than the captured bytes: a frame cut short gets the key of what it holds.
 * An ICMP error belongs to the conversation of the packet it reports on.
 */
packet_flow flow_of(uint32_t link_type, std::span<const uint8_t> frame);

} // namespace pvpcap

#endif // LIBPVPCAP_FLOW_H
