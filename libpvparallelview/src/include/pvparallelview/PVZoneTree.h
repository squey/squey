/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVZONETREE_H
#define PVPARALLELVIEW_PVZONETREE_H

#include <pvkernel/core/inendi_bench.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/PVHardwareConcurrency.h>

#include <inendi/PVSelection.h>
#include <inendi/PVPlotted.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVZoneTreeBase.h>

#include <memory>

#include <boost/array.hpp>
#include <boost/static_assert.hpp>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/scalable_allocator.h>
#include <tbb/task_scheduler_init.h>

constexpr uint32_t TREE_CREATION_GRAINSIZE = 1024;
static_assert(TREE_CREATION_GRAINSIZE % 4 == 0, "TREE_CREATION_GRAINSIZE must be a multiple of 4!");

namespace PVParallelView
{

namespace __impl
{
class TBBMergeTreesTask;
class TBBCreateTreeTask;
class TBBComputeAllocSizeAndFirstElts;
class TBBSelFilterMaxCount;
} // namespace __impl

struct PVZoneProcessing;

class PVZoneTree : public PVZoneTreeBase
{
	friend class __impl::TBBCreateTreeTask;
	friend class __impl::TBBMergeTreesTask;
	friend class __impl::TBBComputeAllocSizeAndFirstElts;
	friend class __impl::TBBSelFilterMaxCount;

  public:
	typedef std::shared_ptr<PVZoneTree> p_type;

  protected:
	typedef std::vector<PVRow, tbb::scalable_allocator<PVRow>> vec_rows_t;
	typedef boost::array<PVRow, NBUCKETS> nbuckets_array_t;
	typedef boost::array<vec_rows_t, NBUCKETS> nbuckets_array_vector_t;
	typedef nbuckets_array_t pdata_array_t;
	typedef nbuckets_array_vector_t pdata_tree_t;
	typedef pdata_tree_t* pdata_tree_pointer_t;

  public:
	struct ProcessData {
		friend class PVZoneTree;
		friend class __impl::TBBCreateTreeTask;
		friend class __impl::TBBMergeTreesTask;
		friend class __impl::TBBComputeAllocSizeAndFirstElts;

		explicit ProcessData(uint32_t n = PVCore::PVHardwareConcurrency::get_physical_core_number())
		    : ntasks(n)
		{
			char* buf = tbb::scalable_allocator<char>().allocate(sizeof(pdata_tree_t) * ntasks +
			                                                     sizeof(pdata_array_t) * ntasks);
			trees = (pdata_tree_t*)buf;
			first_elts = (pdata_array_t*)(trees + ntasks);
			for (uint32_t t = 0; t < ntasks; t++) {
				new (&trees[t]) pdata_tree_t();
				new (&first_elts[t]) pdata_array_t();
			}
		}

		void clear()
		{
			for (uint32_t t = 0; t < ntasks; t++) {
				std::fill(first_elts[t].begin(), first_elts[t].end(), PVROW_INVALID_VALUE);
				for (uint32_t b = 0; b < NBUCKETS; b++) {
					trees[t][b].clear();
				}
			}
		}

		~ProcessData()
		{
			tbb::scalable_allocator<char>().deallocate(
			    (char*)trees, sizeof(pdata_tree_t) * ntasks + sizeof(pdata_array_t) * ntasks);
		}

		pdata_tree_t* trees;
		pdata_array_t* first_elts;
		uint32_t ntasks;
	};

	struct PVBranch {
		PVRow* p;
		size_t count;
	};

  protected:
	struct PVTreeParams {
		// This range is goes from begin (included) to end (*not* included)
		struct PVRange {
			PVRow begin;
			PVRow end;
		};

	  public:
		PVTreeParams(PVZoneProcessing const& zp, PVZoneTree::ProcessData& pdata, uint32_t nrows)
		    : _zp(zp), _pdata(pdata)
		{
			// We compute ranges of row to handle with a min threshold for range size.
			size_t step =
			    (((std::max((nrows + pdata.ntasks - 1) / pdata.ntasks, TREE_CREATION_GRAINSIZE) +
			       3) /
			      4) *
			     4);
			pdata.ntasks = (nrows + step - 1) / step;

			_ranges.resize(pdata.ntasks);
			PVRow cur_r = 0;
			for (uint32_t t = 0; t < pdata.ntasks - 1; t++) {
				_ranges[t].begin = cur_r;
				cur_r += step;
				_ranges[t].end = cur_r;
			}
			_ranges[pdata.ntasks - 1].begin = cur_r;
			_ranges[pdata.ntasks - 1].end = nrows;
		}

	  public:
		inline PVZoneProcessing const& zp() const { return _zp; }
		inline ProcessData& pdata() const { return _pdata; }
		inline const PVRange& range(uint32_t task_num) const { return _ranges[task_num]; }
		inline uint32_t tasks_count() const { return _pdata.ntasks; }

	  private:
		PVZoneProcessing const& _zp;
		ProcessData& _pdata;
		std::vector<PVRange> _ranges;
	};

  public:
	PVZoneTree();
	~PVZoneTree() override
	{
		if (_tree_data) {
			PVCore::PVAlignedAllocator<PVRow, 16>().deallocate(_tree_data, 0);
		}
	}

  public:
	inline void process(PVZoneProcessing const& zp, ProcessData& pdata)
	{
		process_tbb_sse_treeb(zp, pdata);
	}
	inline void process(PVZoneProcessing const& zp) { process_tbb_sse_treeb(zp); }
	inline void filter_by_sel(Inendi::PVSelection const& sel)
	{
		filter_by_sel_tbb_treeb(sel, _sel_elts);
	}
	inline void filter_by_sel_background(Inendi::PVSelection const& sel)
	{
		filter_by_sel_background_tbb_treeb(sel, _bg_elts);
	}

	inline uint32_t get_branch_count(uint32_t branch_id) const { return _treeb[branch_id].count; }

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
	bool operator==(PVZoneTree& zt) const;

  private:
	inline void process_tbb_sse_treeb(PVZoneProcessing const& zp)
	{
		ProcessData pdata;
		process_tbb_sse_treeb(zp, pdata);
	}
	void process_tbb_sse_treeb(PVZoneProcessing const& zp, ProcessData& pdata);

	void filter_by_sel_tbb_treeb(Inendi::PVSelection const& sel, PVRow* buf_elts);
	void filter_by_sel_background_tbb_treeb(Inendi::PVSelection const& sel, PVRow* buf_elts);

  protected:
	PVBranch _treeb[NBUCKETS];
	PVRow* _tree_data;
};

typedef PVZoneTree::p_type PVZoneTree_p;
} // namespace PVParallelView

#endif
