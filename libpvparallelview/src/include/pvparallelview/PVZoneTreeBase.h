/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZONETREEBASE_H
#define PVPARALLELVIEW_PVZONETREEBASE_H

#include <inendi/PVPlotted.h>

#include <pvparallelview/common.h>

namespace PVCore
{
class PVHSVColor;
}

namespace PVParallelView
{

template <size_t Bbits>
class PVBCICode;

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
}

#endif
