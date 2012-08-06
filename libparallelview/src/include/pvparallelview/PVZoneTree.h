/**
 * \file PVZoneTree.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

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
#include <pvkernel/core/PVHardwareConcurrency.h>

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
	typedef nbuckets_array_t pdata_array_t;
	typedef nbuckets_array_vector_t pdata_tree_t;
	typedef pdata_tree_t* pdata_tree_pointer_t;

public:
	struct ProcessData
	{
		friend class PVZoneTree;
		friend class __impl::TBBCreateTreeTask;
		friend class __impl::TBBMergeTreesTask;
		friend class __impl::TBBComputeAllocSizeAndFirstElts;
		friend class __impl::TBBSelFilter;
		friend class __impl::TBBSelRowsFilter;
		friend class __impl::TBBReduceSelElts;

		ProcessData(uint32_t ntasks = PVCore::PVHardwareConcurrency::get_physical_core_number()) : ntasks(ntasks)
		{
			trees = new nbuckets_array_vector_t[ntasks];
			first_elts = new nbuckets_array_t[ntasks];
			sel_elts = new nbuckets_array_t[ntasks];
		}

		// TODO: implement CLEAR methods

		~ProcessData()
		{
			delete [] trees;
			delete [] first_elts;
			delete [] sel_elts;
		}

		pdata_tree_t* trees;
		pdata_array_t* first_elts;
		pdata_array_t* sel_elts;
		uint32_t ntasks;
	};


public://protected:
	struct PVBranch
	{
		PVRow* p;
		size_t count;
	};

	struct PVTreeParams
	{
		struct PVRange
		{
			PVRow begin;
			PVRow end;
		};
	public:
		PVTreeParams(PVZoneProcessing const& zp, PVZoneTree::ProcessData& pdata, uint32_t max_val):
			_zp(zp), _pdata(pdata)
		{
			_ranges = new PVRange[pdata.ntasks];
			PVRow begin = 0;
			PVRow range_size = (((max_val/pdata.ntasks)+4-1)/4)*4;
			for (uint32_t task = 0 ; task < pdata.ntasks-1 ; task++)
			{
				_ranges[task].begin = begin;
				_ranges[task].end = begin+range_size;
				begin += range_size;
			}
			_ranges[pdata.ntasks-1].begin = begin;
			_ranges[pdata.ntasks-1].end = max_val;
		}

		~PVTreeParams()
		{
			delete [] _ranges;
		}
	public:
		//inline PVZoneTree& ztree() const { return _ztree; }
		inline PVZoneProcessing const& zp() const { return _zp; }
		inline ProcessData& pdata() const { return _pdata; }
		inline const PVRange& range(uint32_t task_num) const { return _ranges[task_num]; }
	private:
		PVZoneProcessing const& _zp;
		ProcessData& _pdata;
		PVRange* _ranges;
	};

	struct PVTBBFilterSelParams
		{
		public:
		PVTBBFilterSelParams(PVZoneProcessing const& zp, Picviz::PVSelection const& sel, PVZoneTree::ProcessData& pdata):
				_zp(zp), _sel(sel), _pdata(pdata)
			{ }
		public:
			inline PVZoneProcessing const& zp() const { return _zp; }
			inline ProcessData& pdata() const { return _pdata; }
			inline Picviz::PVSelection const& sel() const { return _sel; }
		private:
			PVZoneProcessing const& _zp;
			Picviz::PVSelection const& _sel;
			ProcessData& _pdata;
		};


public:
	PVZoneTree();
	virtual ~PVZoneTree() { }

public:
	inline void process(PVZoneProcessing const& zp) { process_tbb_sse_treeb(zp); }
	inline void process(PVZoneProcessing const& zp, ProcessData& pdata) { process_tbb_sse_treeb(zp, pdata); }
	inline void filter_by_sel(Picviz::PVSelection const& sel) { filter_by_sel_tbb_treeb(sel); }

	inline void filter_by_sel_new(PVZoneProcessing const& zp, const Picviz::PVSelection& sel, ProcessData& pdata) { filter_by_sel_tbb_treeb_new(zp, sel, pdata); }

	///
	PVBranch* get_treeb() {return _treeb;}
	///

public:
	void process_omp_sse_treeb(PVZoneProcessing const& zp);
	inline void process_tbb_sse_treeb(PVZoneProcessing const& zp) { process_tbb_sse_treeb(zp, _pdata); }
	void process_tbb_sse_treeb(PVZoneProcessing const& zp, ProcessData& pdata);
	void process_tbb_sse_parallelize_on_branches(PVZoneProcessing const& zp);

	void filter_by_sel_omp_treeb(Picviz::PVSelection const& sel);
	void filter_by_sel_tbb_treeb(Picviz::PVSelection const& sel);
	void filter_by_sel_tbb_treeb_new(PVZoneProcessing const& zp, const Picviz::PVSelection& sel, ProcessData& pdata);

private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted, PVRow nrows, PVCol col_a, PVCol col_b);

protected:
	PVBranch _treeb[NBUCKETS];
	PVRow* _tree_data;
	ProcessData _pdata;
};

typedef PVZoneTree::p_type PVZoneTree_p;

}

#endif
