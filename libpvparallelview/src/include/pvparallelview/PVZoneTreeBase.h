/**
 * \file PVZoneTreeBase.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZONETREEBASE_H
#define PVPARALLELVIEW_PVZONETREEBASE_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/common.h>

#include <vector>

namespace PVCore {
class PVHSVColor;
}

namespace PVParallelView {

template <size_t Bbits>
class PVBCICode;


class PVZoneTreeBase
{
protected:
	typedef std::vector<float> pts_t;

public:
	PVZoneTreeBase();
	virtual ~PVZoneTreeBase() { }

public:
	virtual void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted, PVRow nrows, PVCol col_a, PVCol col_b) = 0;
	void display(QString const& name, Picviz::PVPlotted::plotted_table_t const& org_plotted, PVRow nrows, PVCol col_a, PVCol col_b);

	inline uint32_t get_first_elt_of_branch(uint32_t branch_id) const
	{
		return _first_elts[branch_id];
	}

	inline bool branch_valid(uint32_t branch_id) const
	{
		return _first_elts[branch_id] != PVROW_INVALID_VALUE;
	}

	inline const PVRow *get_sel_elts() const
	{
		return _sel_elts;
	}

	size_t browse_tree_bci(PVCore::PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const;
	size_t browse_tree_bci_sel(PVCore::PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const;

	size_t browse_tree_bci_no_sse(PVCore::PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const;
	size_t browse_tree_bci_old(PVCore::PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const;

private:
	size_t browse_tree_bci_from_buffer(const PVRow* elts, PVCore::PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const;

public:
	PVRow DECLARE_ALIGN(16) _first_elts[NBUCKETS];
	PVRow DECLARE_ALIGN(16) _sel_elts[NBUCKETS];
	PVRow DECLARE_ALIGN(16) _bg_elts[NBUCKETS];
};

}

#endif
