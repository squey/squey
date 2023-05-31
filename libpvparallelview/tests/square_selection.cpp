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

#include <pvparallelview/PVSelectionGenerator.h>
#include <pvkernel/core/squey_assert.h>
#include <pvparallelview/PVZoneTree.h>

#include <squey/PVSelection.h>

#include <QRect>
#include <iostream>
#include <chrono>

#ifdef SQUEY_BENCH
constexpr size_t SIZE = 1 << 14;
#else
constexpr size_t SIZE = 1 << 11;
#endif

constexpr size_t width = 100;

int main()
{

	// FIXME: It is allocated on the heap as it is a too big object.
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

	// +1 as we want the whole width but QRect with width == 100 go from 0 to 99 all included
	QRect rect(0, 10, width + 1, 1000); // Skip the 24 last lines and 10 first lines

	Squey::PVSelection sel(SIZE * SIZE);
	sel.select_none();

	auto start = std::chrono::steady_clock::now();

	PVParallelView::PVSelectionGenerator::compute_selection_from_parallel_view_rect(width, *zt,
	                                                                                rect, sel);

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifdef SQUEY_BENCH
	for (size_t i = 0; i < SIZE; i++) {
		for (size_t j = 0; j < SIZE; j++) {
			uint32_t y1 = (plota[i * SIZE + j] >> 22);
			uint32_t y2 = (plotb[i * SIZE + j] >> 22);
			if ((y1 > (size_t)rect.bottomLeft().y() and y2 > (size_t)rect.bottomRight().y()) or
			    (y1 < (size_t)rect.topLeft().y() and y2 < (size_t)rect.topRight().y())) {
				PV_VALID(sel.get_line_fast(i * SIZE + j), false);
			} else {
				PV_VALID(sel.get_line_fast(i * SIZE + j), true);
			}
		}
	}
#endif

	return 0;
}
