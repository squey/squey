/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/inendi_bench.h>
#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/PVBitVisitor.h>
#include <iostream>

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
