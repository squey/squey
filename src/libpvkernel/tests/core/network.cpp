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

#include <pvkernel/core/squey_assert.h>

#include <cstring>

// TODO: add more tests, surely performance stats
const char* ip_text[] = {
    "192.168.23.4", "192.168.23.04",
};

const uint32_t ip_num[] = {
    3232241412, 3232241412,
};

constexpr size_t ip_test_size = sizeof(ip_text) / sizeof(ip_text[0]);
constexpr size_t ip_num_size = sizeof(ip_num) / sizeof(ip_num[0]);

int main(void)
{
	uint32_t n;

	PV_ASSERT_VALID(ip_test_size == ip_num_size);

	for (size_t i = 0; i < ip_test_size; ++i) {
		n = (uint32_t)-1;
		PV_ASSERT_VALID(PVCore::Network::ipv4_aton(ip_text[i], strlen(ip_text[i]), n), "ip",
		                ip_text[i]);

		PV_VALID(n, ip_num[i], "ip", ip_text[i]);
	}

	return 0;
}
