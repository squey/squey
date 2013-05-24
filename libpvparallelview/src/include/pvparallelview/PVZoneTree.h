/**
 * \file PVZoneTree.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZONETREE_H
#define PVPARALLELVIEW_PVZONETREE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/core/PVPODStaticArray.h>
#include <pvkernel/core/PVHSVColor.h>

#include <picviz/PVSelection.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVZoneTreeBase.h>

#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/scalable_allocator.h>
#include <tbb/task_scheduler_init.h>


#define TREE_CREATION_GRAINSIZE 1024
static_assert(TREE_CREATION_GRAINSIZE % 4 == 0, "TREE_CREATION_GRAINSIZE must be a multiple of 4!");

namespace PVParallelView {

namespace __impl {
class TBBMergeTreesTask;
class TBBCreateTreeTask;
class TBBComputeAllocSizeAndFirstElts;
class TBBSelFilter;
class TBBSelFilterMaxCount;
}

class PVZoneProcessing;

class PVZoneTree: public PVZoneTreeBase
{
	friend class __impl::TBBCreateTreeTask;
	friend class __impl::TBBMergeTreesTask;
	friend class __impl::TBBComputeAllocSizeAndFirstElts;
	friend class __impl::TBBSelFilter;
	friend class __impl::TBBSelFilterMaxCount;
	friend class PVSelectionGenerator;

public:
	typedef boost::shared_ptr<PVZoneTree> p_type;

protected:
	typedef std::vector<PVRow, tbb::scalable_allocator<PVRow> > vec_rows_t;
	typedef boost::array<PVRow, NBUCKETS> nbuckets_array_t;
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

		ProcessData(uint32_t n = PVCore::PVHardwareConcurrency::get_physical_core_number()) : ntasks(n)
		{
			char* buf = tbb::scalable_allocator<char>().allocate(sizeof(pdata_tree_t)*ntasks+sizeof(pdata_array_t)*ntasks);
			trees = (pdata_tree_t*) buf;
			first_elts = (pdata_array_t*) (trees+ntasks);
			for (uint32_t t = 0 ; t < ntasks; t++) {
				new (&trees[t]) pdata_tree_t();
				new (&first_elts[t]) pdata_array_t();
			}
		}

		void clear()
		{
			for (uint32_t t = 0 ; t < ntasks; t++) {
				memset(&first_elts[t], PVROW_INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
				for (uint32_t b = 0 ; b < NBUCKETS; b++) {
					trees[t][b].clear();
				}
			}
		}

		~ProcessData()
		{
			tbb::scalable_allocator<char>().deallocate((char*) trees, sizeof(pdata_tree_t)*ntasks+sizeof(pdata_array_t)*ntasks);
		}

		pdata_tree_t* trees;
		pdata_array_t* first_elts;
		uint32_t ntasks;
	};

	struct PVBranch
	{
		PVRow* p;
		size_t count;
	};

protected:
	struct PVTreeParams
	{
		// This range is goes from begin (included) to end (*not* included)
		struct PVRange
		{
			PVRow begin;
			PVRow end;
		};

	public:
		PVTreeParams(PVZoneProcessing const& zp, PVZoneTree::ProcessData& pdata, uint32_t nrows):
			_zp(zp), _pdata(pdata)
		{
			_ranges = new PVRange[pdata.ntasks];
			if (nrows == 0) {
				pdata.ntasks = 0;
				return;
			}


			// Compute the number of tasks according to a minimum grain size
			const uint32_t max_tasks = (nrows+TREE_CREATION_GRAINSIZE-1)/TREE_CREATION_GRAINSIZE;
			const uint32_t ntasks = pdata.ntasks;
			if (max_tasks < ntasks) {
				PVRow cur_r = 0;
				uint32_t t;
				for (t = 0; t < max_tasks-1; t++) {
					_ranges[t].begin = cur_r;
					cur_r += TREE_CREATION_GRAINSIZE;
					_ranges[t].end = cur_r;
				}
				_ranges[t].begin = cur_r;
				_ranges[t].end = nrows;
				pdata.ntasks = max_tasks;
			}
			else {
				PVRow cur_r = 0;
				// The range size is nrows/ntasks, aligned on the next multiple of 4
				PVRow range_size = (((nrows/ntasks)+3)/4)*4;
				uint32_t t;
				for (t = 0; t < ntasks-1; t++) {
					_ranges[t].begin = cur_r;
					cur_r += range_size;
					_ranges[t].end = cur_r;
				}
				_ranges[t].begin = cur_r;
				_ranges[t].end = nrows;
			}
		}

		~PVTreeParams()
		{
			delete [] _ranges;
		}

	public:
		inline PVZoneProcessing const& zp() const { return _zp; }
		inline ProcessData& pdata() const { return _pdata; }
		inline const PVRange& range(uint32_t task_num) const { return _ranges[task_num]; }
		inline uint32_t tasks_count() const { return _pdata.ntasks; }

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
	inline void process(PVZoneProcessing const& zp, ProcessData& pdata) { process_tbb_sse_treeb(zp, pdata); }
	inline void process(PVZoneProcessing const& zp) { process_tbb_sse_treeb(zp); }
	inline void filter_by_sel(Picviz::PVSelection const& sel, const PVRow nrows) { filter_by_sel_tbb_treeb(sel, nrows, _sel_elts); }
	inline void filter_by_sel_background(Picviz::PVSelection const& sel, const PVRow nrows) { filter_by_sel_background_tbb_treeb(sel, nrows, _bg_elts); }

	inline uint32_t get_branch_count(uint32_t branch_id) const
	{
		return _treeb[branch_id].count;
	}

	inline uint32_t get_branch_element(uint32_t branch_id, uint32_t i) const
	{
		return _treeb[branch_id].p[i];
	}

	inline uint32_t set_branch_element(uint32_t branch_id, uint32_t i, uint32_t value)
	{
		return _treeb[branch_id].p[i] = value;
	}

	void dump_branches() const;

	/**
	 * Equality test.
	 *
	 * @param qt the second zoomed zone tree
	 *
	 * @return true if the 2 zone trees have the same structure and the
	 * same content; false otherwise.
	 */
	bool operator==(PVZoneTree &zt) const;

	/**
	 * Save the zone tree into a file.
	 *
	 * @param filename the output filename
	 *
	 * @return true on success; false otherwise and an error is printed.
	 */
	bool dump_to_file(const char *filename) const;

	/**
	 * Create and load a zone tree from a file.
	 *
	 * @param filename the input filename
	 *
	 * @return a zone tree on success; nullptr otherwise and an error is printed.
	 */
	static PVZoneTree *load_from_file(const char *filename);

	/**
	 * Get the number of lines that goes throught a 10-bit plotted value on the right axis of this zone.
	 *
	 * @return the number of lines that goes throught that value
	 */
	inline uint32_t get_right_axis_count(const uint32_t branch_r) const { return get_right_axis_count_seq(branch_r); }

	//inline uint32_t get_right_axis_count(const uint32_t branch_r) const { return get_right_axis_count_seq(branch_r); }

public:
	void process_omp_sse_treeb(PVZoneProcessing const& zp);
	inline void process_tbb_sse_treeb(PVZoneProcessing const& zp) { ProcessData pdata; process_tbb_sse_treeb(zp, pdata); }
	void process_tbb_sse_treeb(PVZoneProcessing const& zp, ProcessData& pdata);
	void process_tbb_sse_parallelize_on_branches(PVZoneProcessing const& zp);

	void filter_by_sel_omp_treeb(Picviz::PVSelection const& sel);
	void filter_by_sel_tbb_treeb(Picviz::PVSelection const& sel, const PVRow nrows, PVRow* buf_elts);
	void filter_by_sel_background_tbb_treeb(Picviz::PVSelection const& sel, const PVRow nrows, PVRow* buf_elts);
	void filter_by_sel_tbb_treeb_new(PVZoneProcessing const& zp, const Picviz::PVSelection& sel);

	uint32_t get_right_axis_count_seq(const uint32_t branch_r) const;

	PVBranch& get_branch(uint32_t branch_id) { return _treeb[branch_id]; }
	PVBranch const& get_branch(uint32_t branch_id) const { return _treeb[branch_id]; }

private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted, PVRow nrows, PVCol col_a, PVCol col_b);

protected:
	PVBranch _treeb[NBUCKETS];
	PVRow* _tree_data;
};

typedef PVZoneTree::p_type PVZoneTree_p;

}

#endif
