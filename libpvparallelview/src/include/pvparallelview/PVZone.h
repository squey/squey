/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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

	inline void filter_by_sel(const Inendi::PVSelection& sel)
	{
		_ztree->filter_by_sel(sel);
		if (_zoomed_ztree->is_initialized()) {
			_zoomed_ztree->compute_min_indexes_sel(sel);
		}
	}

	inline void filter_by_sel_background(const Inendi::PVSelection& sel)
	{
		_ztree->filter_by_sel_background(sel);
	}

  private:
	PVZoneTree_p _ztree;
	PVZoomedZoneTree_p _zoomed_ztree;
};
}

#endif
