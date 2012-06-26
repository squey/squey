#ifndef PVPARALLELVIEW_PVZONETREE_H
#define PVPARALLELVIEW_PVZONETREE_H

#include <pvkernel/core/general.h>
#include <picviz/PVSelection.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/PVZoneProcessing.h>

#include <pvkernel/core/PVPODStaticArray.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneTreeBase.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_scheduler_init.h>


namespace PVParallelView {

namespace __impl {
class TBBMergeTreesTask;
class TBBCreateTreeTask;
class TBBComputeAllocSizeAndFirstElts;
class TBBSelFilter;
class TBBSelRowsFilter;
class TBBReduceSelElts;
}

class PVZoneProcessing;
class PVSelectionSquare;

class PVZoneTree: public PVZoneTreeBase
{
	friend class __impl::TBBCreateTreeTask;
	friend class __impl::TBBMergeTreesTask;
	friend class __impl::TBBComputeAllocSizeAndFirstElts;
	friend class __impl::TBBSelFilter;
	friend class __impl::TBBSelRowsFilter;
	friend class __impl::TBBReduceSelElts;
	friend class PVSelectionSquare;

public:
	typedef boost::shared_ptr<PVZoneTree> p_type;

protected:
	typedef std::vector<PVRow, tbb::scalable_allocator<PVRow> > vec_rows_t;
	typedef PVCore::PVPODStaticArray<PVRow, NBUCKETS, PVROW_INVALID_VALUE> nbuckets_array_t;
	typedef boost::array<vec_rows_t, NBUCKETS> nbuckets_array_vector_t;
	typedef tbb::enumerable_thread_specific<nbuckets_array_t> tls_array_t;
	typedef tbb::enumerable_thread_specific<nbuckets_array_vector_t> tls_tree_t;
	typedef tls_tree_t::pointer tls_tree_pointer_t;

public:
	class ProcessTLS
	{
	public:
		friend class PVZoneTree;
		friend class __impl::TBBCreateTreeTask;
		friend class __impl::TBBMergeTreesTask;
		friend class __impl::TBBComputeAllocSizeAndFirstElts;
		friend class __impl::TBBSelFilter;
		friend class __impl::TBBSelRowsFilter;
		friend class __impl::TBBReduceSelElts;
	protected:
		tls_tree_t _tls_trees;
		tls_array_t _tls_first_elts;
		tls_array_t _tls_sel_elts;
	};


public://protected:
	struct PVBranch
	{
		PVRow* p;
		size_t count;
	};

	struct PVTaskTLSMapper
	{
		PVTaskTLSMapper(uint32_t ntasks) : ntasks(ntasks)
		{
			tls_trees_index = new tls_tree_pointer_t[ntasks];
		}

		~PVTaskTLSMapper()
		{
			delete [] tls_trees_index;
		}

		const uint32_t ntasks;
		mutable tls_tree_pointer_t* tls_trees_index;
	};

	struct PVTreeParams
	{
		struct PVRange
		{
			PVRow begin;
			PVRow end;
		};
	public:
		PVTreeParams(PVZoneProcessing const& zp, PVZoneTree::ProcessTLS& tls, PVTaskTLSMapper& tls_mapper, uint32_t max_val):
			_zp(zp), _tls(tls), _tls_mapper(tls_mapper), _ntasks(tls_mapper.ntasks)
		{
			_ranges = new PVRange[_ntasks];
			PVRow begin = 0;
			PVRow range_size = (((max_val/_ntasks)+4-1)/4)*4;
			for (uint32_t task = 0 ; task < _ntasks-1 ; task++)
			{
				_ranges[task].begin = begin;
				_ranges[task].end = begin+range_size;
				begin += range_size;
			}
			_ranges[_ntasks-1].begin = begin;
			_ranges[_ntasks-1].end = max_val;
		}

		~PVTreeParams()
		{
			delete [] _ranges;
		}
	public:
		//inline PVZoneTree& ztree() const { return _ztree; }
		inline PVZoneProcessing const& zp() const { return _zp; }
		inline ProcessTLS& tls() const { return _tls; }
		inline PVTaskTLSMapper& tls_mapper() const { return _tls_mapper; }
		inline const PVRange& range(uint32_t task_num) const { return _ranges[task_num]; }
		inline uint32_t ntasks() const { return _ntasks; }
	private:
		PVZoneProcessing const& _zp;
		ProcessTLS& _tls;
		PVTaskTLSMapper& _tls_mapper;
		PVRange* _ranges;
		uint32_t _ntasks;
	};

	struct PVTBBFilterSelParams
		{
		public:
		PVTBBFilterSelParams(PVZoneProcessing const& zp, Picviz::PVSelection const& sel, PVZoneTree::ProcessTLS& tls):
				_zp(zp), _sel(sel), _tls(tls)
			{ }
		public:
			inline PVZoneProcessing const& zp() const { return _zp; }
			inline ProcessTLS& tls() const { return _tls; }
			inline Picviz::PVSelection const& sel() const { return _sel; }
		private:
			PVZoneProcessing const& _zp;
			Picviz::PVSelection const& _sel;
			ProcessTLS& _tls;
		};


public:
	PVZoneTree();

public:
	inline void process(PVZoneProcessing const& zp) { process_tbb_sse_treeb(zp); }
	inline void process(PVZoneProcessing const& zp, ProcessTLS& tls) { process_tbb_sse_treeb(zp, tls); }
	inline void filter_by_sel(Picviz::PVSelection const& sel) { filter_by_sel_tbb_treeb(sel); }

	inline void filter_by_sel_new(PVZoneProcessing const& zp, const Picviz::PVSelection& sel, ProcessTLS& tls) { filter_by_sel_tbb_treeb_new(zp, sel, tls); }

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
	void filter_by_sel_tbb_treeb_new(PVZoneProcessing const& zp, const Picviz::PVSelection& sel, ProcessTLS& tls);

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
