#include <pvcore/network.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <QString>
#include <vector>
#include <tbb/tick_count.h>
#include <dnet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

typedef std::vector<QString> list_strings_t;
#define RAND_BYTE (rand()%256)

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " number_ips" << std::endl;
		return 1;
	}

	tbb::tick_count start,end;
	size_t n = atol(argv[1]);

	if (n < 10) n=10;

	srand(time(NULL));

	// Generate a random list of IPs
	list_strings_t ips;
	ips.reserve(n);
	QString pt(".");
	QString str("this is not a valid IP address !!!!!! qskfdpsogjfotjirjoyotjhytrohyntroiyjà(')ç-");
	for (size_t i = 0; i < n; i++) {
		QString ip = QString::number(RAND_BYTE) + pt + QString::number(RAND_BYTE) + pt + QString::number(RAND_BYTE) + pt + QString::number(RAND_BYTE);
		//std::cout << qPrintable(ip) << std::endl;
		ips.push_back(ip);
		// Insert a non valid IP string
		ips.push_back(str);
	}
	n = 2*n;
	float* dst = (float*) malloc(n*sizeof(float));
	memset(dst, 0, n*sizeof(float));

	// And convert them with different ways
	
	// Using PVCore::Network::ipv4_aton (self-made function)
	start = tbb::tick_count::now();
	for (size_t i = 0; i < n; i++) {
		uint32_t ip_n;
		PVCore::Network::ipv4_aton(ips[i], ip_n);
		dst[i] = ip_n;
	}
	end = tbb::tick_count::now();
	std::cout << "PVCore::Network::ipv4_aton: " << (end-start).seconds() << std::endl;

	for (size_t i = 0; i < n; i++) {
		//std::cout << dst[i] << std::endl;
	}

	// Using inet_aton
	start = tbb::tick_count::now();
	for (size_t i = 0; i < n; i++) {
		in_addr_t addr = inet_addr(ips[i].toLatin1().constData());
		uint32_t ip_n = ntohl(addr);
		dst[i] = ip_n;
	}
	end = tbb::tick_count::now();
	std::cout << "inet_aton: " << (end-start).seconds() << std::endl;

	// Using dnet's aton
	start = tbb::tick_count::now();
	for (size_t i = 0; i < n; i++) {
		struct addr adst;
		addr_aton(ips[i].toLatin1().constData(), &adst);
		uint32_t ip_n = ntohl(adst.addr_ip);
		dst[i] = ip_n;
	}
	end = tbb::tick_count::now();
	std::cout << "dnet's aton: " << (end-start).seconds() << std::endl;

	free(dst);
	
	return 0;
}
