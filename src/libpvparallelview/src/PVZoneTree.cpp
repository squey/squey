//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/squey_intrin.h>

#include <squey/PVSelection.h>
#include <squey/PVScaled.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVZoneTree.h>

#include <boost/static_assert.hpp>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_group.h>

#include <omp.h>

#define GRAINSIZE 128

using Squey::PVSelection;

namespace PVParallelView::__impl
{

class TBBCreateTreeTask
{
  public:
	TBBCreateTreeTask(PVParallelView::PVZoneTree::PVTreeParams const& params, uint32_t task_num)
	    : _params(params), _task_num(task_num)
	{
	}

	void operator()() const
	{
		PVParallelView::PVZoneProcessing const& zp = _params.zp();

		const uint32_t* pcol_a = zp.scaled_a;
		const uint32_t* pcol_b = zp.scaled_b;

		PVRow r = _params.range(_task_num).begin;
		PVRow nrows = _params.range(_task_num).end;
		PVRow nrows_sse = (nrows / 4) * 4;

		PVParallelView::PVZoneTree::pdata_array_t& first_elts =
		    _params.pdata().first_elts[_task_num];
		PVParallelView::PVZoneTree::nbuckets_array_vector_t& tree =
		    _params.pdata().trees[_task_num];

		__m128i sse_y1, sse_y2, sse_bcodes;

		for (; r < nrows_sse; r += 4) {
			sse_y1 = _mm_load_si128((const __m128i*)&pcol_a[r]);
			sse_y2 = _mm_load_si128((const __m128i*)&pcol_b[r]);

			sse_y1 = _mm_srli_epi32(sse_y1, 32 - NBITS_INDEX);
			sse_y2 = _mm_srli_epi32(sse_y2, 32 - NBITS_INDEX);
			sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

			uint32_t b0 = _mm_extract_epi32(sse_bcodes, 0);
			if (tree[b0].size() == 0) {
				first_elts[b0] = r + 0;
			}
			tree[b0].push_back(r + 0);

			uint32_t b1 = _mm_extract_epi32(sse_bcodes, 1);
			if (tree[b1].size() == 0) {
				first_elts[b1] = r + 1;
			}
			tree[b1].push_back(r + 1);

			uint32_t b2 = _mm_extract_epi32(sse_bcodes, 2);
			if (tree[b2].size() == 0) {
				first_elts[b2] = r + 2;
			}
			tree[b2].push_back(r + 2);

			uint32_t b3 = _mm_extract_epi32(sse_bcodes, 3);
			if (tree[b3].size() == 0) {
				first_elts[b3] = r + 3;
			}
			tree[b3].push_back(r + 3);
		}
		for (; r < nrows; r++) {
			uint32_t y1 = pcol_a[r];
			uint32_t y2 = pcol_b[r];
			PVParallelView::PVBCode code_b;
			code_b.int_v = 0;
			code_b.s.l = y1 >> (32 - NBITS_INDEX);
			code_b.s.r = y2 >> (32 - NBITS_INDEX);

			if (tree[code_b.int_v].size() == 0) {
				first_elts[code_b.int_v] = r;
			}
			tree[code_b.int_v].push_back(r);
		}
	}

  private:
	PVParallelView::PVZoneTree::PVTreeParams const& _params;
	uint32_t _task_num;
};

class TBBComputeAllocSizeAndFirstElts
{
  public:
	TBBComputeAllocSizeAndFirstElts(PVParallelView::PVZoneTree* ztree,
	                                PVParallelView::PVZoneTree::ProcessData& pdata)
	    : _ztree(ztree), _pdata(pdata), _alloc_size(0)
	{
	}

	TBBComputeAllocSizeAndFirstElts(TBBComputeAllocSizeAndFirstElts& x, tbb::split)
	    : _ztree(x._ztree), _pdata(x._pdata), _alloc_size(0)
	{
	}

  public:
	void operator()(const tbb::blocked_range<size_t>& range)
	{
		for (PVRow b = range.begin(); b != range.end(); ++b) {
			_ztree->_treeb[b].count = 0;
			for (uint32_t task = 0; task < _pdata.ntasks; task++) {
				_ztree->_treeb[b].count += _pdata.trees[task][b].size();
				_ztree->_first_elts[b] =
				    std::min(_ztree->_first_elts[b], _pdata.first_elts[task][b]);
			}
			_alloc_size += (((_ztree->_treeb[b].count + 15) / 16) * 16);
		}
	}

	void join(TBBComputeAllocSizeAndFirstElts const& rhs) { _alloc_size += rhs._alloc_size; }

  public:
	inline size_t alloc_size() const { return _alloc_size; }

  private:
	PVParallelView::PVZoneTree* _ztree;
	PVParallelView::PVZoneTree::ProcessData& _pdata;
	size_t _alloc_size;
};

class TBBMergeTreesTask
{
  public:
	TBBMergeTreesTask(PVParallelView::PVZoneTree* ztree,
	                  PVParallelView::PVZoneTree::PVTreeParams const& params,
	                  uint32_t task_num)
	    : _ztree(ztree), _params(params), _task_num(task_num)
	{
	}

	void operator()() const
	{
		for (PVRow b = _params.range(_task_num).begin; b < _params.range(_task_num).end; ++b) {
			if (_ztree->_treeb[b].count == 0) {
				continue;
			}
			PVRow* cur_branch = _ztree->_treeb[b].p;
			for (uint32_t task = 0; task < _params.pdata().ntasks; task++) {
				PVParallelView::PVZoneTree::vec_rows_t const& branch =
				    _params.pdata().trees[task][b];
				if (branch.size() > 0) {
					memcpy(cur_branch, &branch.at(0), branch.size() * sizeof(PVRow));
					cur_branch += branch.size();
					assert(cur_branch <= _ztree->_treeb[b].p + _ztree->_treeb[b].count);
				}
			}
		}
	}

  private:
	PVParallelView::PVZoneTree* _ztree;
	PVParallelView::PVZoneTree::PVTreeParams const& _params;
	uint32_t _task_num;
};
} // namespace PVParallelView

// PVZoneTree implementation
//

PVParallelView::PVZoneTree::PVZoneTree() : PVZoneTreeBase(), _tree_data(nullptr)
{
}

void PVParallelView::PVZoneTree::process_tbb_sse_treeb(PVZoneProcessing const& zp,
                                                       ProcessData& pdata)
{

	PVRow nrows = zp.size;

	for (uint32_t task = 0; task < pdata.ntasks; task++) {
		std::fill(pdata.first_elts[task].begin(), pdata.first_elts[task].end(),
		          PVROW_INVALID_VALUE);
	}

	BENCH_START(trees);
	tbb::task_group group;
	PVTreeParams create_tree_params(zp, pdata, nrows);
	const size_t ntasks = create_tree_params.tasks_count();
	for (uint32_t t = 0; t < ntasks; t++) {
		group.run(__impl::TBBCreateTreeTask(create_tree_params, t));
	}
	group.wait();
	BENCH_END(trees, "TREES", nrows * 2, sizeof(float), nrows * 2, sizeof(float));

	memset(_treeb, 0, sizeof(PVBranch) * NBUCKETS);

	__impl::TBBComputeAllocSizeAndFirstElts reduce_body(this, pdata);
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, NBUCKETS, GRAINSIZE), reduce_body,
	                     tbb::simple_partitioner());

	if (_tree_data) {
		PVCore::PVAlignedAllocator<PVRow, 4>().deallocate(_tree_data, 0);
	}
	_tree_data = PVCore::PVAlignedAllocator<PVRow, 16>().allocate(reduce_body.alloc_size());

	// Update branch pointer
	PVRow* cur_p = _tree_data;
	for (auto & b : _treeb) {
		if (b.count > 0) {
			b.p = cur_p;
			cur_p += ((b.count + 15) / 16) * 16;
		}
	}

	// Merge trees
	BENCH_START(merge);
	PVTreeParams merge_tree_params(zp, pdata, NBUCKETS);
	for (uint32_t t = 0; t < ntasks; t++) {
		group.run(__impl::TBBMergeTreesTask(this, merge_tree_params, t));
	}
	group.wait();

	BENCH_END(merge, "MERGE", nrows * 2, sizeof(float), nrows * 2, sizeof(float));
}

void PVParallelView::PVZoneTree::filter_by_sel_tbb_treeb(Squey::PVSelection const& sel,
                                                         PVRow* buf_elts)
{
	std::fill_n(buf_elts, NBUCKETS, PVROW_INVALID_VALUE);

	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, GRAINSIZE),
	                  [this, &sel, buf_elts](tbb::blocked_range<size_t> const& br) {
		                  for (PVRow b = br.begin(); b != br.end(); b++) {
			                  PVRow* end = _treeb[b].p + _treeb[b].count;
			                  PVRow* res = std::find_if(_treeb[b].p, end, [&sel](PVRow v) {
				                  return sel.get_line_fast(v);
				              });
			                  if (res != end) {
				                  buf_elts[b] = *res;
			                  }
		                  }
		              },
	                  tbb::simple_partitioner());
}

void PVParallelView::PVZoneTree::filter_by_sel_background_tbb_treeb(Squey::PVSelection const& sel,
                                                                    PVRow* buf_elts)
{
	// returns a zone tree with only the selected events
	Squey::PVSelection::const_pointer sel_buf = sel.get_buffer();
	if (sel_buf == nullptr) {
		// Empty selection
		memcpy(buf_elts, _first_elts, sizeof(PVRow) * NBUCKETS);
		return;
	}
	BENCH_START(subtree2);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, GRAINSIZE),
	                  [&](const tbb::blocked_range<size_t>& range) {
		                  PVRow* buf_elts_ = buf_elts;
		                  PVZoneTree* tree = this;
		                  for (PVRow b = range.begin(); b != range.end(); b++) {
			                  PVRow res = PVROW_INVALID_VALUE;
			                  if (tree->branch_valid(b)) {
				                  const PVRow r = tree->get_first_elt_of_branch(b);
				                  if ((sel_buf[PVSelection::line_index_to_chunk(r)]) &
				                      (1UL << (PVSelection::line_index_to_chunk_bit(r)))) {
					                  res = r;
				                  } else {
					                  for (size_t i = 0; i < tree->_treeb[b].count; i++) {
						                  const PVRow r = tree->_treeb[b].p[i];
						                  if ((sel_buf[PVSelection::line_index_to_chunk(r)]) &
						                      (1UL << (PVSelection::line_index_to_chunk_bit(r)))) {
							                  res = r;
							                  break;
						                  }
					                  }
				                  }
				                  // If nothing from the nu_selection, take the first event (a
				                  // zombie one)
				                  if (res == PVROW_INVALID_VALUE) {
					                  res = r;
				                  }
			                  }
			                  buf_elts_[b] = res;
		                  }
		              });
	BENCH_END(subtree2, "filter_by_sel_background_tbb_treeb", 1, 1, sizeof(PVRow), NBUCKETS);
}

void PVParallelView::PVZoneTree::dump_branches() const
{
	for (size_t i = 0; i < NBUCKETS; i++) {
		if (branch_valid(i) > 0) {
			std::cout << "branch " << i << ": ";
			for (size_t r = 0; r < get_branch_count(i); r++) {
				std::cout << get_branch_element(i, r) << ",";
			}
			std::cout << std::endl;
		}
	}
}

bool PVParallelView::PVZoneTree::operator==(PVParallelView::PVZoneTree& zt) const
{
	for (size_t i = 0; i < NBUCKETS; ++i) {
		if (get_branch_count(i) != zt.get_branch_count(i)) {
			return false;
		}
		for (size_t r = 0; r < get_branch_count(i); ++r) {
			if (get_branch_element(i, r) != zt.get_branch_element(i, r)) {
				return false;
			}
		}
	}

	if (memcmp(_first_elts, zt._first_elts, NBUCKETS * sizeof(PVRow)) != 0) {
		return false;
	} else if (memcmp(_sel_elts, zt._sel_elts, NBUCKETS * sizeof(PVRow)) != 0) {
		return false;
	} else if (memcmp(_bg_elts, zt._bg_elts, NBUCKETS * sizeof(PVRow)) != 0) {
		return false;
	}

	return true;
}
