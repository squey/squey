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
		uint32_t r = i << (32 - 11); // Make sure values are equireparted in the 10 upper bites.
		for (size_t j = 0; j < SIZE; j++) {
			plotb[j * SIZE + i] = plota[i * SIZE + j] = r;
		}
	}

	PVParallelView::PVZoneTree::ProcessData pdata;
	PVParallelView::PVZoneProcessing zp{SIZE * SIZE, plota.data(), plotb.data()};
	zt->process(zp, pdata);

	std::vector<PVRow> rows(NBUCKETS);

	Inendi::PVSelection sel(SIZE * SIZE);
	sel.select_odd(); // Start with 1 as first value thus we check for x % 2 == 0

	auto start = std::chrono::steady_clock::now();

	zt->filter_by_sel(sel);

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifdef INSPECTOR_BENCH
	for (size_t i = 0; i < NBUCKETS; i++) {
		PVRow elt = zt->get_sel_elts()[i];
		PV_ASSERT_VALID(elt % 2 == 0);
	}
#else
	PV_VALID(zt->get_sel_elts()[0], 0U);
#endif

	return 0;
}
