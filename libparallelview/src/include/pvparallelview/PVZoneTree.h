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

namespace __impl {
class TBBCreateTreeNRows; 
class TBBComputeAllocSizeAndFirstElts;
class TBBMergeTrees;
class TBBPF3;
}

class PVZoneProcessing;

class PVZoneTree: public PVZoneTreeBase
{
	friend class __impl::TBBCreateTreeNRows;
	friend class __impl::TBBComputeAllocSizeAndFirstElts;
	friend class __impl::TBBMergeTrees;
	friend class __impl::TBBPF3;

protected:
	struct PVBranch
	{
		PVRow* p;
		size_t count;
	};

	struct PVTBBCreateTreeParams
	{
	public:
		PVTBBCreateTreeParams(PVZoneTree& ztree, PVZoneProcessing const& zp):
			_ztree(ztree), _zp(zp)
		{ }
	public:
		inline PVZoneTree& ztree() const { return _ztree; }
		inline PVZoneProcessing const& zp() const { return _zp; }
	private:
		PVZoneTree& _ztree;
		PVZoneProcessing const& _zp;
	};

protected:
	typedef std::vector<PVRow, tbb::scalable_allocator<PVRow> > vect;
	typedef std::vector<vect, tbb::scalable_allocator<vect> > vectvect;
	typedef tbb::enumerable_thread_specific<vect> TLS;
	typedef tbb::enumerable_thread_specific<vectvect> TLS_List;

public:
	PVZoneTree();

public:
	inline void process(PVZoneProcessing const& zp) { process_tbb_sse_treeb(zp); }
	inline void filter_by_sel(Picviz::PVSelection const& sel) { filter_by_sel_tbb_treeb(sel); }

public:
	void process_omp_sse_treeb(PVZoneProcessing const& zp);
	void process_tbb_sse_treeb(PVZoneProcessing const& zp);
	void process_tbb_sse_parallelize_on_branches(PVZoneProcessing const& zp);

	void filter_by_sel_omp_treeb(Picviz::PVSelection const& sel);
	void filter_by_sel_tbb_treeb(Picviz::PVSelection const& sel);

private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted, PVRow nrows, PVCol col_a, PVCol col_b);

protected:
	PVBranch* _treeb;
	PVRow* _tree_data;
	TLS_List tls_trees;
	TLS tls_first_elts;
};

}

#endif
