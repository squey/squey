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

PVParallelView::PVZoneTreeBase::PVZoneTreeBase()
{
	memset(_first_elts, PVROW_INVALID_VALUE, sizeof(PVRow) * NBUCKETS);
	memset(_sel_elts, PVROW_INVALID_VALUE, sizeof(PVRow) * NBUCKETS);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci(PVCore::PVHSVColor const* colors,
                                                       PVBCICode<NBITS_INDEX>* codes) const
{
	return browse_tree_bci_from_buffer(_bg_elts, colors, codes);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_sel(PVCore::PVHSVColor const* colors,
                                                           PVBCICode<NBITS_INDEX>* codes) const
{
	return browse_tree_bci_from_buffer(_sel_elts, colors, codes);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_from_buffer(
    const PVRow* elts, PVCore::PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const
{
	size_t idx_code = 0;

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
