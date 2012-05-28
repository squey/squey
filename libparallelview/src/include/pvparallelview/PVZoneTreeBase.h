#ifndef PVPARALLELVIEW_PVZONETREEBASE_H
#define PVPARALLELVIEW_PVZONETREEBASE_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/common.h>

#include <vector>

namespace PVParallelView {

class PVBCICode;
class PVHSVColor;

class PVZoneTreeBase
{
protected:
	typedef std::vector<float> pts_t;

public:
	PVZoneTreeBase();

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


	size_t browse_tree_bci(PVHSVColor const* colors, PVBCICode* codes) const;
	size_t browse_tree_bci_sel(PVHSVColor const* colors, PVBCICode* codes) const;

	size_t browse_tree_bci_no_sse(PVHSVColor const* colors, PVBCICode* codes) const;
	size_t browse_tree_bci_old(PVHSVColor const* colors, PVBCICode* codes) const;

private:
	size_t browse_tree_bci_from_buffer(const PVRow* elts, PVHSVColor const* colors, PVBCICode* codes) const;

public:
	PVRow DECLARE_ALIGN(16) _first_elts[NBUCKETS];
	PVRow DECLARE_ALIGN(16) _sel_elts[NBUCKETS];
};

}

#endif
