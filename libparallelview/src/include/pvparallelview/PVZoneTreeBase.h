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
	void set_trans_plotted(Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols);
	virtual void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted) = 0;
	void display(QString const& name, Picviz::PVPlotted::plotted_table_t const& org_plotted);
	inline uint32_t get_plotted_value(PVRow r, PVCol c) const { return (*_plotted)[c*_nrows_aligned + r]; }
	inline uint32_t const* get_plotted_col(PVCol c) const { return &((*_plotted)[c*_nrows_aligned]); }

	inline uint32_t get_first_elt_of_branch(uint32_t branch_id) const
	{
		return _first_elts[branch_id];
	}

	inline bool branch_valid(uint32_t branch_id) const
	{
		return _first_elts[branch_id] != PVROW_INVALID_VALUE;
	}

	size_t browse_tree_bci_no_sse(PVHSVColor* colors, PVBCICode* codes);
	size_t browse_tree_bci_old(PVHSVColor* colors, PVBCICode* codes);
	size_t browse_tree_bci(PVHSVColor* colors, PVBCICode* codes);

public:
	PVRow DECLARE_ALIGN(16) _first_elts[NBUCKETS];
	PVRow DECLARE_ALIGN(16) _sel_elts[NBUCKETS];

	Picviz::PVPlotted::uint_plotted_table_t const* _plotted;
	PVCol _col_a;
	PVCol _col_b;
	PVRow _nrows;
	PVCol _ncols;
	PVRow _nrows_aligned;
};

}

#endif
