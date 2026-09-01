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

#include <pvkernel/core/squey_bench.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvhwloc.h>

#include <squey/PVSelection.h>
#include <squey/PVScaled.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVZoneTreeBase.h>

#include <memory>

//! Smallest share of rows worth giving a task of its own.
constexpr uint32_t TREE_CREATION_GRAINSIZE = 1024;

namespace PVParallelView
{

struct PVZoneProcessing;

class PVZoneTree : public PVZoneTreeBase
{
  public:
	typedef std::shared_ptr<PVZoneTree> p_type;

  public:
	/**
	 * How many tasks a build may be split over.
	 *
	 * This used to carry the buffers a build worked in -- one growable list per
	 * bucket per task, tens of megabytes each -- which is why callers hold on to
	 * it between zones. Sorting the rows by partition first left nothing worth
	 * keeping: a build now counts into one small array per task, sized in
	 * kilobytes, and allocates it where it uses it.
	 */
	struct ProcessData {
		explicit ProcessData(uint32_t n = pvhwloc::core_count())
		    : _max_tasks(std::max<uint32_t>(1, n))
		{
		}

		//! Kept for callers that used to have to hand the buffers back.
		void clear() {}

		inline uint32_t max_tasks() const { return _max_tasks; }

	  private:
		uint32_t _max_tasks;
	};

	struct PVBranch {
		PVRow* p;
		size_t count;
	};

  public:
	PVZoneTree();
	~PVZoneTree() override
	{
		if (_tree_data) {
			// The alignment plays no part in freeing -- deallocate just calls free --
			// and asking for 16 here only makes the compiler doubt the pointer.
			PVCore::PVAlignedAllocator<PVRow, 4>().deallocate(_tree_data, 0);
		}
	}

  public:
	inline void process(PVZoneProcessing const& zp, ProcessData& pdata)
	{
		process_tbb_sse_treeb(zp, pdata);
	}
	inline void process(PVZoneProcessing const& zp) { process_tbb_sse_treeb(zp); }
	inline void filter_by_sel(Squey::PVSelection const& sel)
	{
		filter_by_sel_tbb_treeb(sel, _sel_elts);
	}
	inline void filter_by_sel_background(Squey::PVSelection const& sel)
	{
		filter_by_sel_background_tbb_treeb(sel, _bg_elts);
	}

	inline uint32_t get_branch_count(uint32_t branch_id) const { return _treeb[branch_id].count; }

	inline uint32_t get_branch_element(uint32_t branch_id, uint32_t i) const
	{
		return _treeb[branch_id].p[i];
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

	void filter_by_sel_tbb_treeb(Squey::PVSelection const& sel, PVRow* buf_elts);
	void filter_by_sel_background_tbb_treeb(Squey::PVSelection const& sel, PVRow* buf_elts);

  protected:
	PVBranch _treeb[NBUCKETS];
	PVRow* _tree_data = nullptr;
	//! Rows the store was allocated for, so a rebuild at the same size keeps it.
	size_t _tree_data_size = 0;
};

typedef PVZoneTree::p_type PVZoneTree_p;
} // namespace PVParallelView

#endif
