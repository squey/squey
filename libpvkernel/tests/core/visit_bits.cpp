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

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/inendi_bench.h>
#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/PVBitVisitor.h>

#include <iostream>
#include <vector>
#include <numeric>

void check_bit_visitor(int* bits_ref, size_t size)
{
	uint64_t v = 0;
	for (size_t i = 0; i < size; i++) {
		v |= 1ULL << (bits_ref[i]);
	}

	std::vector<int> bits;
	PVCore::PVBitVisitor::visit_bits(v, [&](unsigned int b) { bits.push_back(b); });

	PV_ASSERT_VALID(size == bits.size());
	PV_ASSERT_VALID(std::equal(bits.begin(), bits.end(), bits_ref));
}

int main()
{
	std::cout << "Check bit visitor" << std::endl;
	int bits[64];
	std::iota(std::begin(bits), std::end(bits), 0);
	check_bit_visitor(bits, sizeof(bits) / sizeof(int));
	std::cout << "done" << std::endl;

	return 0;
}
