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

#include <pvkernel/core/squey_intrin.h>

#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVZoneTree.h>

#include <cassert>
#include <cmath>

PVParallelView::PVZoneTreeBase::PVZoneTreeBase()
{
	memset(_first_elts, PVROW_INVALID_VALUE, sizeof(PVRow) * NBUCKETS);
	memset(_sel_elts, PVROW_INVALID_VALUE, sizeof(PVRow) * NBUCKETS);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci(PVCore::PVHSVColor const* colors,
                                                       PVBCICode<NBITS_INDEX>* codes,
                                                       float line_opacity) const
{
	return browse_tree_bci_from_buffer(_bg_elts, _bg_counts, colors, codes, line_opacity);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_sel(PVCore::PVHSVColor const* colors,
                                                           PVBCICode<NBITS_INDEX>* codes,
                                                           float line_opacity) const
{
	return browse_tree_bci_from_buffer(_sel_elts, _sel_counts, colors, codes, line_opacity);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_from_buffer(
    const PVRow* elts,
    std::vector<uint32_t> const& counts,
    PVCore::PVHSVColor const* colors,
    PVBCICode<NBITS_INDEX>* codes,
    float line_opacity) const
{
	size_t idx_code = 0;

	if (line_opacity < 1.f) {
		// The counts follow the list of occupied buckets, which is walked however
		// full it is.
		const bool counted = counts.size() == _occupied_branches.size();
		const float log_transparency = std::log1p(-line_opacity);

		for (size_t i = 0; i < _occupied_branches.size(); i++) {
			const uint32_t b = _occupied_branches[i];
			const PVRow r = elts[b];
			if (r == PVROW_INVALID_VALUE) {
				continue;
			}

			// A bucket with a row of the layer counts that row at least: none means
			// it was not counted.
			const uint32_t rows = counted ? counts[i] : 0;
			const float opacity = rows > 0 ? -std::expm1(rows * log_transparency) : 1.f;
			const auto opacity8 = static_cast<uint8_t>(opacity * 255.f + .5f);
			if (opacity8 == 0) {
				continue;
			}

			PVBCICode<NBITS_INDEX> bci;
			bci.int_v = r | ((uint64_t)b << 32);
			bci.s.color = colors[r].h();
			bci.set_opacity(opacity8);
			codes[idx_code] = bci;
			idx_code++;
		}
		return idx_code;
	}

	// Only occupied buckets can name a row here: filter_by_sel and
	// filter_by_sel_background leave every other entry of `elts` alone. Reading
	// the bucket numbers back costs a buffer of its own, so this stops paying
	// once nearly every bucket is occupied and the sweep it replaces no longer
	// wastes a branch on anything; measured, the two meet at about 15/16 full.
	if (_occupied_branches.size() < (NBUCKETS / 16) * 15) {
		for (const uint32_t b : _occupied_branches) {
			const PVRow r = elts[b];
			if (r != PVROW_INVALID_VALUE) {
				PVBCICode<NBITS_INDEX> bci;
				bci.int_v = r | ((uint64_t)b << 32);
				bci.s.color = colors[r].h();
				codes[idx_code] = bci;
				idx_code++;
			}
		}
		return idx_code;
	}

	for (uint32_t b = 0; b < NBUCKETS; b++) {
		const PVRow r = elts[b];
		if (r != PVROW_INVALID_VALUE) {
			PVBCICode<NBITS_INDEX> bci;
			bci.int_v = r | ((uint64_t)b << 32);
			bci.s.color = colors[r].h();
			codes[idx_code] = bci;
			idx_code++;
		}
	}

	return idx_code;
}
