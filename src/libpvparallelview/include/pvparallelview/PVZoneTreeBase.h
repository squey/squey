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

#include <vector>

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

	/**
	 * How many selected rows each occupied bucket holds, in the order of
	 * occupied_branches(), as of the last PVZoneTree::filter_by_sel asked to count
	 * them; zeros until then.
	 */
	inline std::vector<uint32_t> const& get_sel_counts() const { return _sel_counts; }

	/**
	 * As get_sel_counts(), for PVZoneTree::filter_by_sel_background: the rows of
	 * its selection in each bucket, or all of them where it selects none and the
	 * bucket's first row stands for them as a zombie.
	 */
	inline std::vector<uint32_t> const& get_bg_counts() const { return _bg_counts; }

	/**
	 * The buckets that hold at least one row, in ascending order.
	 *
	 * There are a million buckets and a zone rarely fills more than a handful of
	 * them, so everything that used to sweep the whole range -- generating BCI
	 * codes, filtering by selection -- walks this instead. Filled in by
	 * PVZoneTree::process; empty until then, which is correct, as an unbuilt tree
	 * has nothing to walk.
	 */
	inline std::vector<uint32_t> const& occupied_branches() const { return _occupied_branches; }

	/**
	 * Encode the lines of the background as BCI codes, one per bucket holding a
	 * row of it; browse_tree_bci_sel does the same for the selection.
	 *
	 * @param line_opacity the opacity of the line of a single row, in ]0, 1].
	 * Below 1, the lines are drawn by density (see PVBCIDrawingBackend::render):
	 * each code carries the opacity of the n rows of its bucket laid over one
	 * another, 1 - (1 - line_opacity)^n, from the counts of the last filtering
	 * (see get_bg_counts() and get_sel_counts()). A bucket the filtering did not
	 * count is drawn opaque; one too faint to show at all is left out.
	 *
	 * @return the number of codes written to @p codes.
	 */
	size_t browse_tree_bci(PVCore::PVHSVColor const* colors,
	                       PVBCICode<NBITS_INDEX>* codes,
	                       float line_opacity = 1.f) const;
	size_t browse_tree_bci_sel(PVCore::PVHSVColor const* colors,
	                           PVBCICode<NBITS_INDEX>* codes,
	                           float line_opacity = 1.f) const;

  private:
	size_t browse_tree_bci_from_buffer(const PVRow* elts,
	                                   std::vector<uint32_t> const& counts,
	                                   PVCore::PVHSVColor const* colors,
	                                   PVBCICode<NBITS_INDEX>* codes,
	                                   float line_opacity) const;

  public:
	PVRow DECLARE_ALIGN(16) _first_elts[NBUCKETS];
	PVRow DECLARE_ALIGN(16) _sel_elts[NBUCKETS];
	PVRow DECLARE_ALIGN(16) _bg_elts[NBUCKETS];

  protected:
	std::vector<uint32_t> _occupied_branches;

	// Sized along with _occupied_branches rather than when counted: a rendering
	// may read them while a selection is being counted, and must not see them
	// reallocated.
	std::vector<uint32_t> _sel_counts;
	std::vector<uint32_t> _bg_counts;
};
} // namespace PVParallelView

#endif
