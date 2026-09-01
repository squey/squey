//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// A zone tree encodes the row of each occupied bucket as a BCI code: the row, the
// bucket its line runs between, and the row's colour. The occupied buckets are walked
// from the list the tree keeps of them, or all swept once nearly all are occupied;
// whichever walk encodes a row, its code has to carry that row's own bucket and colour.

#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/squey_assert.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVZoneTreeBase.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace
{

using code_t = PVParallelView::PVBCICode<NBITS_INDEX>;

//! A tree whose rows are placed by hand, and the list of occupied buckets with them.
struct placed_tree : PVParallelView::PVZoneTreeBase {
	placed_tree() { std::memset(_bg_elts, 0xFF, sizeof(_bg_elts)); }

	void place(uint32_t bucket, PVRow row)
	{
		_bg_elts[bucket] = row;
		const auto at = std::lower_bound(_occupied_branches.begin(), _occupied_branches.end(), bucket);
		_occupied_branches.insert(at, bucket);
	}
};

void check_codes(const placed_tree& tree,
                 const std::vector<PVCore::PVHSVColor>& colors,
                 size_t rows,
                 const char* walk)
{
	code_t* codes = code_t::allocate_codes(NBUCKETS);
	const size_t count = tree.browse_tree_bci(colors.data(), codes);
	PV_ASSERT_VALID(count == rows, "walk", walk, "codes", count, "rows", rows);

	for (size_t i = 0; i < count; ++i) {
		const code_t& code = codes[i];
		const PVRow row = code.s.idx;
		const uint32_t bucket = code.s.l | (code.s.r << NBITS_INDEX);
		PV_ASSERT_VALID(tree._bg_elts[bucket] == row, "walk", walk, "row", row,
		                "encoded in bucket", bucket);
		PV_ASSERT_VALID(code.s.color == colors[row].h(), "walk", walk, "row", row,
		                "encoded in colour", uint32_t(code.s.color));
	}
	code_t::free_codes(codes);
}

} // namespace

int main()
{
	// A few rows, walked from the list of occupied buckets.
	{
		auto tree = std::make_unique<placed_tree>();
		const std::vector<std::pair<uint32_t, PVRow>> placed = {
		    {1024 * 7 + 40, 3},  {1024 * 7 + 41, 0},   {1024 * 7 + 42, 5},  {1024 * 7 + 43, 1},
		    {1024 * 300 + 8, 2}, {1024 * 300 + 10, 4}, {1024 * 300 + 11, 6}};
		std::vector<PVCore::PVHSVColor> colors(placed.size());
		for (const auto& [bucket, row] : placed) {
			tree->place(bucket, row);
			colors[row] = PVCore::PVHSVColor(uint8_t(10 + 20 * row));
		}
		check_codes(*tree, colors, placed.size(), "list");
	}

	// Every bucket occupied, which is swept.
	{
		auto tree = std::make_unique<placed_tree>();
		std::vector<PVCore::PVHSVColor> colors(NBUCKETS);
		for (uint32_t bucket = 0; bucket < uint32_t(NBUCKETS); ++bucket) {
			const PVRow row = PVRow(NBUCKETS - 1 - bucket);
			tree->place(bucket, row);
			colors[row] = PVCore::PVHSVColor(uint8_t(row % 251));
		}
		check_codes(*tree, colors, NBUCKETS, "sweep");
	}

	std::cout << "every BCI code carries its own row's bucket and colour" << std::endl;
	return 0;
}
