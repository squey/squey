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
