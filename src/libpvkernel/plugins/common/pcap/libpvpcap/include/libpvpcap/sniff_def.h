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

#ifndef __SNIFF_DEF_H__
#define __SNIFF_DEF_H__

#include <pcap.h>
#ifndef _WIN32
#include <sys/socket.h>
#include <arpa/inet.h> // for inet_ntoa()
#include <net/ethernet.h>
#include <netinet/ip_icmp.h> //Provides declarations for icmp header
#include <netinet/udp.h>     //Provides declarations for udp header
#include <netinet/tcp.h>     //Provides declarations for tcp header
#include <netinet/ip.h>      //Provides declarations for ip header
#include <netinet/ip6.h>
#endif

#define IP_V(ip) (((ip)->ip_vhl) >> 4)

// IP header
struct sniff_ip {
	u_char ip_vhl;                 /* version << 4 | header length >> 2 */
	u_char ip_tos;                 /* type of service */
	u_short ip_len;                /* total length */
	u_short ip_id;                 /* identification */
	u_short ip_off;                /* fragment offset field */
#define IP_RF 0x8000               /* reserved fragment flag */
#define IP_DF 0x4000               /* dont fragment flag */
#define IP_MF 0x2000               /* more fragments flag */
#define IP_OFFMASK 0x1fff          /* mask for fragmenting bits */
	u_char ip_ttl;                 /* time to live */
	u_char ip_p;                   /* protocol */
	u_short ip_sum;                /* checksum */
	struct in_addr ip_src, ip_dst; /* source and dest address */
};
#define IP_HL(ip) (((ip)->ip_vhl) & 0x0f)
#define IP_V(ip) (((ip)->ip_vhl) >> 4)

// TCP header
struct sniff_tcp {
	u_short th_sport; /* source port */
	u_short th_dport; /* destination port */
	uint32_t th_seq;   /* sequence number */
	uint32_t th_ack;   /* acknowledgement number */
	u_char th_offx2;  /* data offset, rsvd */
#define TH_OFF(th) (((th)->th_offx2 & 0xf0) >> 4)
	u_char th_flags;
#define TH_FIN 0x01
#define TH_SYN 0x02
#define TH_RST 0x04
#define TH_PUSH 0x08
#define TH_ACK 0x10
#define TH_URG 0x20
#define TH_ECE 0x40
#define TH_CWR 0x80
#define TH_FLAGS (TH_FIN | TH_SYN | TH_RST | TH_ACK | TH_URG | TH_ECE | TH_CWR)
	u_short th_win; /* window */
	u_short th_sum; /* checksum */
	u_short th_urp; /* urgent pointer */
};

// UDP header
struct sniff_udp {
	uint16_t sport; /* source port */
	uint16_t dport; /* destination port */
	uint16_t udp_length;
	uint16_t udp_sum; /* checksum */
};

#ifdef _WIN32

#pragma pack(push, 1) // Ensure correct memory alignment

struct ethhdr {
    uint8_t  h_dest[6];  // Destination MAC address
    uint8_t  h_source[6]; // Source MAC address
    uint16_t h_proto;    // EtherType (e.g., 0x0800 for IPv4, 0x86DD for IPv6)
};

struct ip6_hdr {
    union {
        struct {
            uint32_t ip6_un1_flow; // 4-bit Version + 8-bit Traffic Class + 20-bit Flow Label
            uint16_t ip6_un1_plen; // Payload Length
            uint8_t  ip6_un1_nxt;  // Next Header
            uint8_t  ip6_un1_hlim; // Hop Limit
        } ip6_un1;
        uint8_t ip6_vfc; // Version and Traffic Class (quick access)
    };
    struct in6_addr ip6_src; // IPv6 Source Address
    struct in6_addr ip6_dst; // IPv6 Destination Address
};

#pragma pack(pop) // Restore default alignment
#endif // _WIN32

#endif // __SNIFF_DEF_H__
