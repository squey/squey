#include <stdio.h>
#include <stdlib.h>

#ifdef WIN32
#include <pcap.h>
#else
#include <pcap/pcap.h>
#endif

int main(void)
{
  pcap_t *pcaph;
  u_char *packet;
  struct pcap_pkthdr pheader;
  struct timeval packet_tv;
  time_t t;

  char *str;

        pcaph = pcap_open_offline("zeus.pcap", NULL);
	if (!pcaph) {
		fprintf(stderr, "Cannot open PCAP file\n");
		return NULL;
	}
        packet = (u_char *)pcap_next(pcaph, &pheader);
        while (packet) {
		packet_tv = pheader.ts;
		t = packet_tv.tv_sec;
		printf("tv_sec=%d\n", t);
		str = ctime(&t);
		str[strlen(str) - 1] = '\0';
		printf("time string='%s'\n", str);
		exit(0);
	}


  return 0;
}
