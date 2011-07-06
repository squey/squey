#include <pcap/pcap.h>
#include <dnet.h>
#include <unistd.h>

char errbuf[PCAP_ERRBUF_SIZE];

int main(int argv, char **argc)
{
	pcap_t *pcaph;
	struct pcap_pkthdr pheader;
	u_char *packet;

	struct eth_hdr *ether;
	struct ip_hdr *ip;
	struct tcp_hdr *tcp;
	struct udp_hdr *udp;

	char *ipsource;
	char *ipdest;

	int i = 0;

	if (argv < 2) {
		fprintf(stderr, "Syntax: pcap2csv file.pcap > file.csv\n");
		exit(1);
	}
	
	pcaph = pcap_open_offline(argc[1], errbuf);

	packet = (u_char *)pcap_next(pcaph, &pheader);
	while (packet) {
		ether = (struct eth_hdr *)(packet);

		if(ntohs(ether->eth_type) == ETH_TYPE_IP) {
			ip = (struct ip_hdr *)(packet + ETH_HDR_LEN);
			ipsource = ip_ntoa(&ip->ip_src);
			ipdest = ip_ntoa(&ip->ip_dst);
			printf("%s,%s,", ipsource, ipdest);
			/* printf("IP Source: %s; Dest: %s\n", ipsource, ipdest); */

			if (ip->ip_p == IP_PROTO_TCP) {
				tcp = (struct tcp_hdr *)(packet + ETH_HDR_LEN + IP_HDR_LEN);
				printf("tcp,%d,%d,", tcp->th_sport, tcp->th_dport);
				printf("%d\n", ip->ip_len);
			}
			if (ip->ip_p == IP_PROTO_UDP) {
				udp = (struct udp_hdr *)(packet + ETH_HDR_LEN + IP_HDR_LEN);
				printf("udp,%d,%d,", udp->uh_sport, udp->uh_dport);
				printf("%d\n", ip->ip_len);
			}
		}


		packet = (u_char *)pcap_next(pcaph, &pheader);
		i++;
	}

	/* printf("There are %d packets\n", i); */

	return 0;
}
