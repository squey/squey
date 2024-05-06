/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVZONETREEBASE_H
#define PVPARALLELVIEW_PVZONETREEBASE_H

#include <squey/PVScaled.h>

#include <pvparallelview/common.h>

namespace PVCore
{
class PVHSVColor;
} // namespace PVCore

namespace PVParallelView
{

template <size_t Bbits>
struct PVBCICode;

class PVZoneTreeBase
{
  public:
	PVZoneTreeBase();
	virtual ~PVZoneTreeBase() {}

  public:
	inline uint32_t get_first_elt_of_branch(uint32_t branch_id) const
	{
		return _first_elts[branch_id];
	}

	inline bool branch_valid(uint32_t branch_id) const
	{
		return _first_elts[branch_id] != PVROW_INVALID_VALUE;
	}

	inline const PVRow* get_sel_elts() const { return _sel_elts; }

	inline const PVRow* get_bg_elts() const { return _bg_elts; }

	size_t browse_tree_bci(PVCore::PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const;
	size_t browse_tree_bci_sel(PVCore::PVHSVColor const* colors,
	                           PVBCICode<NBITS_INDEX>* codes) const;

  private:
	size_t browse_tree_bci_from_buffer(const PVRow* elts,
	                                   PVCore::PVHSVColor const* colors,
	                                   PVBCICode<NBITS_INDEX>* codes) const;

  public:
	PVRow DECLARE_ALIGN(16) _first_elts[NBUCKETS];
	PVRow DECLARE_ALIGN(16) _sel_elts[NBUCKETS];
	PVRow DECLARE_ALIGN(16) _bg_elts[NBUCKETS];
};
} // namespace PVParallelView

#endif
