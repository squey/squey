/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/network.h>

#include <cctype> // for isspace

static uint32_t powui(uint32_t base, uint32_t n)
{
	uint32_t ret = 1;
	for (uint32_t i = 0; i < n; i++) {
		ret *= base;
	}
	return ret;
}

bool PVCore::Network::ipv4_aton(const char* str, size_t n, uint32_t& ret)
{
	if ((n > 15) || (n == 0)) {
		return false;
	}

	uint32_t ndots = 0;
	uint32_t nip = 0;
	int32_t cur_d = 2;
	size_t i = 0;
	while ((i < n) && (std::isspace(str[i]) != 0)) {
		i++;
	}
	char prev_c = 0;
	ret = 0;
	for (; i < n; i++) {
		const char c = str[i];
		if (c == '.') {
			if (prev_c == '.') {
				return false;
			}
			ndots++;
			nip /= powui(10, cur_d + 1);
			if (nip > 255 || ndots > 3) {
				return false;
			}
			ret |= nip << ((4 - ndots) * 8);
			nip = 0;
			cur_d = 2;
		} else {
			if ((c < '0') || (c > '9')) {
				if (prev_c == '.') {
					return false;
				}
				break;
			}
			if (cur_d < 0) {
				return false;
			}
			nip += (static_cast<uint32_t>((c - '0'))) * powui(10, cur_d);
			cur_d--;
		}
		prev_c = c;
	}
	nip /= powui(10, cur_d + 1);
	if (nip > 255) {
		return false;
	}
	ret |= nip;
	// Check trailing characters
	for (; i < n; i++) {
		if (std::isspace(str[i]) == 0) {
			return false;
		}
	}
	return ndots == 3;
}
