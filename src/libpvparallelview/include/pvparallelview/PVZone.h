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

#ifndef PVPARALLELVIEW_PVZONE_H
#define PVPARALLELVIEW_PVZONE_H

#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/common.h>

namespace PVParallelView
{

/**
 * Gather zone tree and zoomed zone tree of a Zone.
 */
class PVZone
{
  public:
	PVZone()
	    : _ztree(new PVZoneTree())
	    , _zoomed_ztree(new PVZoomedZoneTree(_ztree->get_sel_elts(), _ztree->get_bg_elts()))
	{
	}

  public:
	PVZoneTree& ztree() { return *_ztree; }
	PVZoneTree const& ztree() const { return *_ztree; }

	PVZoomedZoneTree& zoomed_ztree() { return *_zoomed_ztree; }
	PVZoomedZoneTree const& zoomed_ztree() const { return *_zoomed_ztree; }

	inline void filter_by_sel(const Squey::PVSelection& sel)
	{
		_ztree->filter_by_sel(sel);
		if (_zoomed_ztree->is_initialized()) {
			_zoomed_ztree->compute_min_indexes_sel(sel);
		}
	}

	inline void filter_by_sel_background(const Squey::PVSelection& sel)
	{
		_ztree->filter_by_sel_background(sel);
	}

  private:
	PVZoneTree_p _ztree;
	PVZoomedZoneTree_p _zoomed_ztree;
};
} // namespace PVParallelView

#endif
