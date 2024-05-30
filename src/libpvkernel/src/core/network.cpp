//
// MIT License
//
// © ESI Group, 2015
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
