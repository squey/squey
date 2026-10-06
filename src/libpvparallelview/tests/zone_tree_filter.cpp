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
#include <pvkernel/core/squey_assert.h>
#include <pvparallelview/PVBCode.h>

#include <chrono>
#include <iostream>
#include <string>

#ifdef SQUEY_BENCH
constexpr size_t SCALING_SIZE = 10000; // Means X ** 2 total lines
#else
// Number of line on each scaling axe (use all combination between these values)
constexpr size_t SCALING_SIZE = (size_t)1 << 11;
#endif

namespace
{

uint32_t selected_rows(PVParallelView::PVZoneTree const& zt,
                       uint32_t bucket,
                       Squey::PVSelection const& sel)
{
	uint32_t selected = 0;
	for (uint32_t i = 0; i < zt.get_branch_count(bucket); ++i) {
		selected += sel.get_line(zt.get_branch_element(bucket, i));
	}
	return selected;
}

/**
 * Filter asking for the rows of each bucket to be counted, and check the counts
 * against the rows of the bucket tested one by one.
 */
void check_counts(PVParallelView::PVZoneTree& zt, Squey::PVSelection const& sel, std::string name)
{
	zt.filter_by_sel(sel, true);
	zt.filter_by_sel_background(sel, true);

	const std::vector<uint32_t>& occupied = zt.occupied_branches();
	PV_ASSERT_VALID(zt.get_sel_counts().size() == occupied.size(), "selection", name,
	                "selection counts", zt.get_sel_counts().size(), "occupied buckets",
	                occupied.size());
	PV_ASSERT_VALID(zt.get_bg_counts().size() == occupied.size(), "selection", name,
	                "background counts", zt.get_bg_counts().size(), "occupied buckets",
	                occupied.size());

	for (size_t i = 0; i < occupied.size(); ++i) {
		const uint32_t bucket = occupied[i];
		const uint32_t selected = selected_rows(zt, bucket, sel);
		PV_ASSERT_VALID(zt.get_sel_counts()[i] == selected, "selection", name, "bucket", bucket,
		                "counted", zt.get_sel_counts()[i], "selected", selected);

		// A bucket the selection keeps none of shows its first row, a zombie
		// standing for all of them.
		const uint32_t background = selected > 0 ? selected : zt.get_branch_count(bucket);
		PV_ASSERT_VALID(zt.get_bg_counts()[i] == background, "selection", name, "bucket", bucket,
		                "counted in the background", zt.get_bg_counts()[i], "expected",
		                background);
	}
}

} // namespace

/**
 * Check ZoneTree building from two scaled axes and its bucket creations.
 */
int main()
{

	std::unique_ptr<PVParallelView::PVZoneTree> zt(new PVParallelView::PVZoneTree());

	std::vector<uint32_t> plota(SCALING_SIZE * SCALING_SIZE);
	std::vector<uint32_t> plotb(SCALING_SIZE * SCALING_SIZE);

	// Generate scaling to have equireparted line on both sides.
	for (size_t i = 0; i < SCALING_SIZE; i++) {
		uint32_t r = i << (32 - 11); // Make sure values are equireparted in the 10 upper bites.
		for (size_t j = 0; j < SCALING_SIZE; j++) {
			plotb[j * SCALING_SIZE + i] = plota[i * SCALING_SIZE + j] = r;
		}
	}

	PVParallelView::PVZoneTree::ProcessData pdata;
	PVParallelView::PVZoneProcessing zp{SCALING_SIZE * SCALING_SIZE, plota.data(), plotb.data()};
	zt->process(zp, pdata);

	Squey::PVSelection sel(SCALING_SIZE * SCALING_SIZE);
	sel.select_odd(); // Start with 1 as first value thus we check for x % 2 == 0

	auto start = std::chrono::steady_clock::now();

	zt->filter_by_sel(sel);

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifdef SQUEY_BENCH
	for (size_t i = 0; i < NBUCKETS; i++) {
		PVRow elt = zt->get_sel_elts()[i];
		PV_ASSERT_VALID(elt % 2 == 0);
	}
#else
	PV_VALID(zt->get_sel_elts()[0], 0U);

	// Not counted unless asked for.
	PV_ASSERT_VALID(zt->get_sel_counts()[0] == 0, "uncounted rows", zt->get_sel_counts()[0]);

	check_counts(*zt, sel, "odd rows");

	Squey::PVSelection all(SCALING_SIZE * SCALING_SIZE);
	all.select_all();
	check_counts(*zt, all, "every row");

	Squey::PVSelection none(SCALING_SIZE * SCALING_SIZE);
	none.select_none();
	check_counts(*zt, none, "no row");

	// Some buckets with none of their rows selected, others with one or two.
	Squey::PVSelection some(SCALING_SIZE * SCALING_SIZE);
	some.select_none();
	for (PVRow r = 0; r < SCALING_SIZE * SCALING_SIZE; r += 7) {
		some.set_bit_fast(r);
	}
	check_counts(*zt, some, "one row in seven");
#endif

	return 0;
}
