#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvparallelview/PVBCode.h>

#include <chrono>
#include <iostream>

#ifdef INSPECTOR_BENCH
constexpr size_t SIZE = 10000; // Means X ** 2 total lines
#else
// Number of line on each plotting axe (use all combination between these values)
constexpr size_t SIZE = 1 << 11;
#endif

constexpr size_t significant_bits = 10;

/**
 * Check ZoneTree building from two plotted axes and its bucket creations.
 */
int main()
{

	std::unique_ptr<PVParallelView::PVZoneTree> zt(new PVParallelView::PVZoneTree());

	std::vector<uint32_t> plota(SIZE * SIZE);
	std::vector<uint32_t> plotb(SIZE * SIZE);

	// Generate plotting to have equireparted line on both sides.
	for (size_t i = 0; i < SIZE; i++) {
		uint32_t r =
		    i << (32 - significant_bits); // Make sure values are equireparted in the 10 upper bits.
		for (size_t j = 0; j < SIZE; j++) {
			plotb[j * SIZE + i] = plota[i * SIZE + j] = r;
		}
	}

	PVParallelView::PVZoneTree::ProcessData pdata;
	PVParallelView::PVZoneProcessing zp{SIZE * SIZE, plota.data(), plotb.data()};

	auto start = std::chrono::steady_clock::now();

	zt->process(zp, pdata);

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	for (size_t i = 0; i < SIZE * SIZE; i++) {
		uint32_t pos =
		    (plotb[i] >> (32 - significant_bits) << 10 | (plota[i] >> (32 - significant_bits)));
		PVParallelView::PVBCode b_code;
		b_code.int_v = 0;
		b_code.s.l = (plota[i] >> (32 - significant_bits));
		b_code.s.r = (plotb[i] >> (32 - significant_bits));

		// Check bit ordering for packed structure is as expected
		PV_VALID(b_code.int_v, pos);
		PV_ASSERT_VALID(b_code.int_v < NBUCKETS);

		// Check Number of row per branches with pow2 equireparted values.
		PV_VALID(static_cast<size_t>(zt->get_branch_count(b_code.int_v)),
		         (SIZE * SIZE + NBUCKETS - 1) / NBUCKETS);
	}
#endif

	// Check elements are in the correct bucket
	// ie : b_code value is the same as recomputed one.
	for (uint32_t i = 0; i < NBUCKETS; i++) {
		size_t count = zt->get_branch_count(i);
		for (size_t j = 0; j < count; j++) {
			uint32_t elt = zt->get_branch_element(i, j);
			PVParallelView::PVBCode b_code_line;
			b_code_line.int_v = 0;
			b_code_line.s.l = (plota[elt] >> (32 - significant_bits));
			b_code_line.s.r = (plotb[elt] >> (32 - significant_bits));
			PV_VALID(b_code_line.int_v, i);
		}
	}

	return 0;
}
