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

#ifndef SQUEY_TESTS_PCAP_CAPTURES_H
#define SQUEY_TESTS_PCAP_CAPTURES_H

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

/**
 * Frames and captures written by the tests, rather than kept as files: what each test reads
 * is then in front of whoever reads the test.
 */
namespace pcap_captures
{

using bytes = std::vector<uint8_t>;

inline void put16(bytes& b, uint16_t v)
{
	b.push_back(v >> 8);
	b.push_back(v & 0xff);
}

inline void put32(bytes& b, uint32_t v)
{
	put16(b, v >> 16);
	put16(b, v & 0xffff);
}

inline bytes operator+(bytes a, const bytes& b)
{
	a.insert(a.end(), b.begin(), b.end());
	return a;
}

using ipv4_address = std::array<uint8_t, 4>;
using ipv6_address = std::array<uint8_t, 16>;

inline ipv6_address ipv6(uint16_t a, uint16_t b, uint16_t last)
{
	ipv6_address address{};
	address[0] = a >> 8;
	address[1] = a & 0xff;
	address[2] = b >> 8;
	address[3] = b & 0xff;
	address[14] = last >> 8;
	address[15] = last & 0xff;
	return address;
}

enum : uint8_t { TCP = 6, UDP = 17, ICMP = 1, ICMPV6 = 58 };
enum : uint8_t { FIN = 0x01, SYN = 0x02, PSH = 0x08, ACK = 0x10 };

inline bytes tcp(uint16_t source,
                 uint16_t destination,
                 uint32_t seq,
                 uint32_t ack,
                 uint8_t flags,
                 const std::string& payload = {})
{
	bytes b;
	put16(b, source);
	put16(b, destination);
	put32(b, seq);
	put32(b, ack);
	b.push_back(5 << 4);
	b.push_back(flags);
	put16(b, 65535);
	put16(b, 0);
	put16(b, 0);
	b.insert(b.end(), payload.begin(), payload.end());
	return b;
}

inline bytes udp(uint16_t source, uint16_t destination, const bytes& payload)
{
	bytes b;
	put16(b, source);
	put16(b, destination);
	put16(b, static_cast<uint16_t>(8 + payload.size()));
	put16(b, 0);
	return b + payload;
}

/**
 * @param fragment offset of the fragment in bytes, and whether more follow: a datagram
 *        not fragmented has neither
 */
inline bytes ipv4(const ipv4_address& source,
                  const ipv4_address& destination,
                  uint8_t protocol,
                  const bytes& payload,
                  uint16_t id = 1,
                  uint16_t offset = 0,
                  bool more = false)
{
	bytes b;
	b.push_back(0x45);
	b.push_back(0);
	put16(b, static_cast<uint16_t>(20 + payload.size()));
	put16(b, id);
	put16(b, static_cast<uint16_t>((more ? 0x2000 : 0) | (offset / 8)));
	b.push_back(64);
	b.push_back(protocol);
	put16(b, 0);
	b.insert(b.end(), source.begin(), source.end());
	b.insert(b.end(), destination.begin(), destination.end());
	uint32_t sum = 0;
	for (size_t i = 0; i < 20; i += 2) {
		sum += (b[i] << 8) | b[i + 1];
	}
	while (sum >> 16) {
		sum = (sum & 0xffff) + (sum >> 16);
	}
	b[10] = (~sum >> 8) & 0xff;
	b[11] = ~sum & 0xff;
	return b + payload;
}

inline bytes ipv6(const ipv6_address& source,
                  const ipv6_address& destination,
                  uint8_t next,
                  const bytes& payload)
{
	bytes b;
	put32(b, 0x60000000);
	put16(b, static_cast<uint16_t>(payload.size()));
	b.push_back(next);
	b.push_back(64);
	b.insert(b.end(), source.begin(), source.end());
	b.insert(b.end(), destination.begin(), destination.end());
	return b + payload;
}

inline bytes mac(uint8_t last)
{
	return {0x02, 0x00, 0x00, 0x00, 0x00, last};
}

//! An Ethernet frame, tagged by as many 802.1Q headers as there are VLAN given.
inline bytes ethernet(uint16_t type, const bytes& payload, std::vector<uint16_t> vlans = {})
{
	bytes b = mac(2) + mac(1);
	for (size_t i = 0; i < vlans.size(); i++) {
		put16(b, i == 0 and vlans.size() > 1 ? 0x88a8 : 0x8100);
		put16(b, vlans[i]);
	}
	put16(b, type);
	return b + payload;
}

//! A frame of a Linux "any" capture.
inline bytes linux_cooked(uint16_t type, const bytes& payload)
{
	bytes b;
	put16(b, 0);
	put16(b, 1);
	put16(b, 6);
	b = b + mac(1);
	put16(b, 0);
	put16(b, type);
	return b + payload;
}

struct frame {
	uint32_t seconds;
	uint32_t microseconds;
	bytes data;
};

//! A pcap file in the byte order of this machine, with microsecond timestamps.
inline void
write_pcap(const std::string& path, uint32_t link_type, const std::vector<frame>& frames)
{
	std::ofstream out(path, std::ios::binary);
	const auto put = [&](auto v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
	put(uint32_t(0xa1b2c3d4));
	put(uint16_t(2));
	put(uint16_t(4));
	put(int32_t(0));
	put(uint32_t(0));
	put(uint32_t(262144));
	put(link_type);
	for (const frame& f : frames) {
		put(f.seconds);
		put(f.microseconds);
		put(uint32_t(f.data.size()));
		put(uint32_t(f.data.size()));
		out.write(reinterpret_cast<const char*>(f.data.data()), std::streamsize(f.data.size()));
	}
}

} // namespace pcap_captures

#endif // SQUEY_TESTS_PCAP_CAPTURES_H
