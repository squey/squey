#ifndef PVPARALLELVIEW_PVZONETREENOALLOC_H
#define PVPARALLELVIEW_PVZONETREENOALLOC_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVPODTree.h>

#include <pvparallelview/PVZoneTreeBase.h>

#include <tbb/enumerable_thread_specific.h>

namespace PVParallelView {

class PVZoneTreeNoAlloc: public PVZoneTreeBase
{
public:
	typedef PVCore::PVPODTree<uint32_t, uint32_t, NBUCKETS> Tree;
	typedef std::vector<PVRow, tbb::scalable_allocator<PVRow> > vect;
	typedef tbb::enumerable_thread_specific<vect> TLS;
public:
	PVZoneTreeNoAlloc(PVCol col_a, PVCol col_b):
		_col_a(col_a), _col_b(col_b)
	{ }
public:
	void process_sse();
	void process_omp_sse();

	void filter_by_sel_omp(Picviz::PVSelection const& sel);
	void filter_by_sel_tbb(Picviz::PVSelection const& sel);

	size_t browse_tree_bci_by_sel(PVHSVColor* colors, PVBCICode* codes, Picviz::PVSelection const& sel);

private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted);

public://private:
	Tree _tree;
	PVCol _col_a;
	PVCol _col_b;
};

}

#endif
