#ifndef PVPARALLELVIEW_PVZONETREE_H
#define PVPARALLELVIEW_PVZONETREE_H

#include <pvkernel/core/general.h>
#include <picviz/PVSelection.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVHSVColor.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneTreeBase.h>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <tbb/enumerable_thread_specific.h>

namespace PVParallelView {

/******************************************************************************
 *
 * PVParallelView::PVZoneTree
 *
 *****************************************************************************/
class PVZoneTree: public PVZoneTreeBase
{
	struct PVBranch
	{
		PVRow* p;
		size_t count;
	};

public:
	typedef std::vector<PVRow, tbb::scalable_allocator<PVRow> > vect;
	typedef std::vector<vect, tbb::scalable_allocator<vect> > vectvect;
	typedef tbb::enumerable_thread_specific<vect> TLS;
	typedef tbb::enumerable_thread_specific<vectvect> TLS_List;

public:
	PVZoneTree(PVCol col_a, PVCol col_b):
		_col_a(col_a), _col_b(col_b)
	{ }

public:
	void process_omp_sse_treeb();
	void process_tbb_sse_treeb();
	void process_tbb_sse_parallelize_on_branches();

	PVZoneTree* filter_by_sel_omp_treeb(Picviz::PVSelection const& sel);
	PVZoneTree* filter_by_sel_tbb_treeb(Picviz::PVSelection const& sel);
private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted);
public://private:
	PVCol _col_a;
	PVCol _col_b;
	PVBranch* _treeb;
	PVRow* _tree_data;
	TLS_List tls_trees;
	TLS tls_first_elts;
};

}

#endif
