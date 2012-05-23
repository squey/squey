#ifndef PVPARALLELVIEW_PVZONETREE_H
#define PVPARALLELVIEW_PVZONETREE_H

#include <pvkernel/core/general.h>
#include <picviz/PVSelection.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVHSVColor.h>

#include <pvkernel/core/PVPODStaticArray.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneTreeBase.h>

#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <tbb/enumerable_thread_specific.h>

namespace PVParallelView {

namespace __impl {
class TBBCreateTreeNRows; 
class TBBComputeAllocSizeAndFirstElts;
class TBBMergeTrees;
class TBBSelFilter;
}

class PVZoneProcessing;

class PVZoneTree: public PVZoneTreeBase
{
	friend class __impl::TBBCreateTreeNRows;
	friend class __impl::TBBComputeAllocSizeAndFirstElts;
	friend class __impl::TBBMergeTrees;
	friend class __impl::TBBSelFilter;

public:
	typedef boost::shared_ptr<PVZoneTree> p_type;

protected:
	typedef std::vector<PVRow, tbb::scalable_allocator<PVRow> > vec_rows_t;
	typedef PVCore::PVPODStaticArray<PVRow, NBUCKETS, PVROW_INVALID_VALUE> nbuckets_array_t;
	typedef boost::array<vec_rows_t, NBUCKETS> nbuckets_array_vector_t;
	typedef tbb::enumerable_thread_specific<nbuckets_array_t> tls_array_t;
	typedef tbb::enumerable_thread_specific<nbuckets_array_vector_t> tls_tree_t;

public:
	class ProcessTLS
	{
		friend class PVZoneTree;
		friend class __impl::TBBCreateTreeNRows;
		friend class __impl::TBBComputeAllocSizeAndFirstElts;
		friend class __impl::TBBMergeTrees;
		friend class __impl::TBBSelFilter;
	protected:
		tls_tree_t _tls_trees;
		tls_array_t _tls_first_elts;
	};


public://protected:
	struct PVBranch
	{
		PVRow* p;
		size_t count;
	};

	struct PVTBBCreateTreeParams
	{
	public:
		PVTBBCreateTreeParams(PVZoneProcessing const& zp, PVZoneTree::ProcessTLS& tls):
			_zp(zp), _tls(tls)
		{ }
	public:
		//inline PVZoneTree& ztree() const { return _ztree; }
		inline PVZoneProcessing const& zp() const { return _zp; }
		inline ProcessTLS& tls() const { return _tls; }
	private:
		PVZoneProcessing const& _zp;
		ProcessTLS& _tls;
	};


public:
	PVZoneTree();

public:
	inline void process(PVZoneProcessing const& zp) { process_tbb_sse_treeb(zp); }
	inline void process_tls(PVZoneProcessing const& zp, ProcessTLS tls) { process_tbb_sse_treeb(zp, tls); }
	inline void filter_by_sel(Picviz::PVSelection const& sel) { filter_by_sel_tbb_treeb(sel); }

	///
	PVBranch* get_treeb() {return _treeb;}
	///

public:
	void process_omp_sse_treeb(PVZoneProcessing const& zp);
	inline void process_tbb_sse_treeb(PVZoneProcessing const& zp) { process_tbb_sse_treeb(zp, _tls); }
	void process_tbb_sse_treeb(PVZoneProcessing const& zp, ProcessTLS& tls);
	void process_tbb_sse_parallelize_on_branches(PVZoneProcessing const& zp);

	void filter_by_sel_omp_treeb(Picviz::PVSelection const& sel);
	void filter_by_sel_tbb_treeb(Picviz::PVSelection const& sel);

private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted, PVRow nrows, PVCol col_a, PVCol col_b);

protected:
	PVBranch _treeb[NBUCKETS];
	PVRow* _tree_data;
	ProcessTLS _tls;
};

typedef PVZoneTree::p_type PVZoneTree_p;

}

#endif
