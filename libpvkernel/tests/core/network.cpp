/**
 * \file network.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/network.h>
#include <stdio.h>
#include <arpa/inet.h>

#include <string>

#include <pvkernel/core/picviz_assert.h>

// TODO: add more tests, surely performance stats
const char* ip_text[] = {
	"192.168.23.4",
	"192.168.23.04",
};

const uint32_t ip_num[] = {
	3232241412,
	3232241412,
};

constexpr size_t ip_test_size = sizeof(ip_text) / sizeof(ip_text[0]);
constexpr size_t ip_num_size = sizeof(ip_num) / sizeof(ip_num[0]);

int main(void)
{
	uint32_t n;

	PV_ASSERT_VALID(ip_test_size == ip_num_size);

	for (size_t i = 0; i < ip_test_size; ++i) {
		n = (uint32_t)-1;
		PV_ASSERT_VALID(PVCore::Network::ipv4_aton(QString(ip_text[i]), n), "ip", ip_text[i]);

		PV_VALID(n, ip_num[i], "ip", ip_text[i]);

		const std::string res = PVCore::Network::ipv4_ntoa(ntohl(n));
		PV_VALID(res, std::string(ip_text[0]));
	}

	return 0;
}
