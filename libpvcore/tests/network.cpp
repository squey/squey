#include <pvcore/network.h>

#include <stdio.h>

int main(void)
{
  PVCore::Network network("192.168.23.4");
  
  if (network.is_ip_addr()){
    printf("IP Addr\n");
  }


  return 0;
}
