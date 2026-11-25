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

// Both directions of a conversation get the same key, whatever carries them.
//
// The import gives the frames of a conversation to the same tshark, which reassembles it as
// one tshark reading the whole capture would. A key that differs between the two directions
// sends them to different processes: TCP streams that no longer reassemble, HTTP responses
// matched to no request, and two stream ids for one connection. The key of an IPv6 packet
// used to read its protocol from its source address, and that of a tagged frame its ports
// four bytes too early, which these check against.

#include <pvkernel/core/squey_assert.h>

#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/flow.h"
#include "pcap_captures.h"

#include <iostream>

using namespace pcap_captures;

namespace
{

constexpr uint32_t ETHERNET = 1;
constexpr uint32_t RAW = 101;
constexpr uint32_t LINUX_SLL = 113;

const ipv4_address client4{10, 0, 0, 1};
const ipv4_address server4{192, 168, 1, 2};
// Their second bytes differ, which the protocol was read from.
const ipv6_address client6 = ipv6(0x2001, 0x0db8, 1);
const ipv6_address server6 = ipv6(0x2a00, 0x1450, 2);

pvpcap::flow_key key(uint32_t link, const bytes& frame)
{
	return pvpcap::flow_of(link, frame).key;
}

void check_same(const char* what, const pvpcap::flow_key& a, const pvpcap::flow_key& b)
{
	std::cout << what << std::endl;
	PV_ASSERT_VALID(a == b, "the two directions of", std::string(what));
}

void check_different(const char* what, const pvpcap::flow_key& a, const pvpcap::flow_key& b)
{
	std::cout << what << std::endl;
	PV_ASSERT_VALID(not(a == b), "two conversations", std::string(what));
}

} // namespace

int main()
{
	const bytes request4 = ipv4(client4, server4, TCP, tcp(40000, 80, 1, 0, SYN));
	const bytes response4 = ipv4(server4, client4, TCP, tcp(80, 40000, 7, 2, SYN | ACK));
	const bytes request6 = ipv6(client6, server6, TCP, tcp(40000, 80, 1, 0, SYN));
	const bytes response6 = ipv6(server6, client6, TCP, tcp(80, 40000, 7, 2, SYN | ACK));

	check_same("IPv4 over Ethernet", key(ETHERNET, ethernet(0x0800, request4)),
	           key(ETHERNET, ethernet(0x0800, response4)));
	check_same("IPv6 over Ethernet", key(ETHERNET, ethernet(0x86dd, request6)),
	           key(ETHERNET, ethernet(0x86dd, response6)));
	check_same("IPv4 over 802.1Q", key(ETHERNET, ethernet(0x0800, request4, {10})),
	           key(ETHERNET, ethernet(0x0800, response4, {10})));
	check_same("IPv6 over 802.1ad", key(ETHERNET, ethernet(0x86dd, request6, {100, 10})),
	           key(ETHERNET, ethernet(0x86dd, response6, {100, 10})));
	check_same("IPv4 over Linux cooked", key(LINUX_SLL, linux_cooked(0x0800, request4)),
	           key(LINUX_SLL, linux_cooked(0x0800, response4)));
	check_same("raw IPv6", key(RAW, request6), key(RAW, response6));

	// The ports are read, and tell conversations between the same hosts apart.
	const pvpcap::flow_key tagged = key(ETHERNET, ethernet(0x0800, request4, {10}));
	PV_VALID(tagged.port_a == 80 or tagged.port_b == 80, true);
	PV_VALID(tagged.port_a == 40000 or tagged.port_b == 40000, true);
	const pvpcap::flow_key v6 = key(ETHERNET, ethernet(0x86dd, request6));
	PV_VALID(int(v6.protocol), int(TCP));
	check_different(
	    "other ports", key(ETHERNET, ethernet(0x0800, request4)),
	    key(ETHERNET, ethernet(0x0800, ipv4(client4, server4, TCP, tcp(40001, 80, 1, 0, SYN)))));
	check_different(
	    "the same ports over UDP", key(ETHERNET, ethernet(0x0800, request4)),
	    key(ETHERNET, ethernet(0x0800, ipv4(client4, server4, UDP, udp(40000, 80, {})))));

	// The fragments of a datagram are known by the datagram: the first holds the ports.
	const bytes datagram = udp(5353, 53, bytes(64, 0xaa));
	const bytes first(datagram.begin(), datagram.begin() + 32);
	const bytes second(datagram.begin() + 32, datagram.end());
	const pvpcap::packet_flow head =
	    pvpcap::flow_of(ETHERNET, ethernet(0x0800, ipv4(client4, server4, UDP, first, 7, 0, true)));
	const pvpcap::packet_flow tail =
	    pvpcap::flow_of(ETHERNET, ethernet(0x0800, ipv4(client4, server4, UDP, second, 7, 32)));
	std::cout << "fragments" << std::endl;
	PV_VALID(head.fragment and head.first_fragment, true);
	PV_VALID(tail.fragment and not tail.first_fragment, true);
	PV_VALID(head.datagram == tail.datagram and head.datagram_id == tail.datagram_id, true);
	PV_VALID(head.key == key(ETHERNET, ethernet(0x0800, ipv4(client4, server4, UDP, datagram))),
	         true);

	// An ICMP error belongs to the conversation of the packet it quotes.
	const bytes quoted = ipv4(client4, server4, UDP, udp(5353, 53, {}));
	bytes unreachable = {3, 3, 0, 0, 0, 0, 0, 0};
	unreachable.insert(unreachable.end(), quoted.begin(), quoted.begin() + 28);
	check_same("ICMP port unreachable",
	           key(ETHERNET, ethernet(0x0800, ipv4(server4, client4, ICMP, unreachable))),
	           key(ETHERNET, ethernet(0x0800, quoted)));

	// Frames cut short get the key of what they hold, without reading past them.
	for (size_t size = 0; size <= request6.size() + 14; size++) {
		const bytes frame = ethernet(0x86dd, request6);
		pvpcap::flow_of(ETHERNET, bytes(frame.begin(), frame.begin() + size));
	}

	return 0;
}
