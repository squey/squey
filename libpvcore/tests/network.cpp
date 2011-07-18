#include <pvcore/network.h>

#include <stdio.h>

int main(void)
{
	uint32_t n;
  
  if (PVCore::Network::ipv4_aton("192.168.23.4", n)) {
    printf("IP Addr\n");
	return 0;
  }

  return 1;
}
