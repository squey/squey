/**
 * \file PVZoneTree.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVAlignedBlockedRange.h>
#include <pvkernel/core/PVPODTree.h>
#include <pvkernel/core/PVHSVColor.h>

#include <picviz/PVSelection.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVZoneTree.h>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_group.h>

#include <omp.h>

#define GRAINSIZE 100

using Picviz::PVSelection;

namespace PVParallelView { namespace __impl {

class TBBCreateTreeTask
{
public:
	TBBCreateTreeTask(
		PVParallelView::PVZoneTree::PVTreeParams const& params,
		uint32_t task_num
	)
		: _params(params),
		  _task_num(task_num)
	{}

    void operator()()
	{
		PVParallelView::PVZoneProcessing const& zp = _params.zp();

		PVCol col_a = zp.col_a();
		PVCol col_b = zp.col_b();

		const uint32_t* pcol_a = zp.get_plotted_col(col_a);
		const uint32_t* pcol_b = zp.get_plotted_col(col_b);

		PVRow r = _params.range(_task_num).begin;
		PVRow nrows = _params.range(_task_num).end;
		PVRow nrows_sse = (nrows/4)*4;

		PVParallelView::PVZoneTree::pdata_array_t& first_elts = _params.pdata().first_elts[_task_num];
		PVParallelView::PVZoneTree::nbuckets_array_vector_t& tree = _params.pdata().trees[_task_num];

		__m128i sse_y1, sse_y2, sse_bcodes;

		for (; r < nrows_sse; r += 4) {
			sse_y1 = _mm_load_si128((const __m128i*) &pcol_a[r]);
			sse_y2 = _mm_load_si128((const __m128i*) &pcol_b[r]);

			sse_y1 = _mm_srli_epi32(sse_y1, 32-NBITS_INDEX);
			sse_y2 = _mm_srli_epi32(sse_y2, 32-NBITS_INDEX);
			sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

			uint32_t b0 = _mm_extract_epi32(sse_bcodes, 0);
			if (tree[b0].size() == 0) {
				first_elts[b0] = r+0;
			}
			tree[b0].push_back(r+0);

			uint32_t b1 = _mm_extract_epi32(sse_bcodes, 1);
			if (tree[b1].size() == 0) {
				first_elts[b1] = r+1;
			}
			tree[b1].push_back(r+1);

			uint32_t b2 = _mm_extract_epi32(sse_bcodes, 2);
			if (tree[b2].size() == 0) {
				first_elts[b2] = r+2;
			}
			tree[b2].push_back(r+2);

			uint32_t b3 = _mm_extract_epi32(sse_bcodes, 3);
			if (tree[b3].size() == 0) {
				first_elts[b3] = r+3;
			}
			tree[b3].push_back(r+3);
		}
		for (; r < nrows; r++) {
			uint32_t y1 = pcol_a[r];
			uint32_t y2 = pcol_b[r];
			PVParallelView::PVBCode code_b;
			code_b.int_v = 0;
			code_b.s.l = y1 >> (32-NBITS_INDEX);
			code_b.s.r = y2 >> (32-NBITS_INDEX);

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
	TBBComputeAllocSizeAndFirstElts (
		PVParallelView::PVZoneTree* ztree, PVParallelView::PVZoneTree::ProcessData& pdata
	) :
		_ztree(ztree),
		_pdata(pdata),
		_alloc_size(0)
	{ }

	TBBComputeAllocSizeAndFirstElts(TBBComputeAllocSizeAndFirstElts& x, tbb::split) :
		_ztree(x._ztree),
		_pdata(x._pdata),
		_alloc_size(0)
	{ }

public:
	void operator() (const tbb::blocked_range<size_t>& range) const
	{
		for (PVRow b = range.begin(); b != range.end(); ++b) {
			_ztree->_treeb[b].count = 0;
			for (uint32_t task = 0 ; task < _pdata.ntasks; task++) {
				_ztree->_treeb[b].count += _pdata.trees[task][b].size();
				_ztree->_first_elts[b] = picviz_min(_ztree->_first_elts[b], _pdata.first_elts[task][b]);
			}
			_alloc_size += (((_ztree->_treeb[b].count + 15) / 16) * 16);
		}
	}

	void join(TBBComputeAllocSizeAndFirstElts& rhs)
	{
		_alloc_size += rhs._alloc_size;
	}

public:
	inline size_t alloc_size() const { return _alloc_size; }

private:
	PVParallelView::PVZoneTree* _ztree;
	PVParallelView::PVZoneTree::ProcessData& _pdata;
	mutable size_t _alloc_size;
};

class TBBMergeTreesTask
{
public:
	TBBMergeTreesTask (PVParallelView::PVZoneTree* ztree, PVParallelView::PVZoneTree::PVTreeParams const& params, uint32_t task_num):
		_ztree(ztree), _params(params), _task_num(task_num)
	{ }

	void operator()()
	{
		for (PVRow b = _params.range(_task_num).begin; b < _params.range(_task_num).end; ++b) {
			if (_ztree->_treeb[b].count == 0) {
				continue;
			}
			PVRow* cur_branch = _ztree->_treeb[b].p;
			for (uint32_t task = 0 ; task < _params.pdata().ntasks ; task++) {
				PVParallelView::PVZoneTree::vec_rows_t const& branch = _params.pdata().trees[task][b];
				if (branch.size() > 0) {
					memcpy(cur_branch, &branch.at(0), branch.size()*sizeof(PVRow));
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

class TBBSelFilter {
public:
	TBBSelFilter (
		PVParallelView::PVZoneTree* tree,
		const Picviz::PVSelection::const_pointer sel_buf
	) :
		_tree(tree),
		_sel_buf(sel_buf)
	{
	}

	void operator() (const tbb::blocked_range<size_t>& range) const
	{
		for (PVRow b = range.begin(); b != range.end(); ++b) {
			PVRow res = PVROW_INVALID_VALUE;
			if (_tree->branch_valid(b)) {
				const PVRow r = _tree->get_first_elt_of_branch(b);
				if ((_sel_buf[PVSelection::line_index_to_chunk(r)]) & (1UL<<(PVSelection::line_index_to_chunk_bit(r)))) {
					res = r;
				}
				else {
					for (size_t i=0; i< _tree->_treeb[b].count; i++) {
						const PVRow r = _tree->_treeb[b].p[i];
						if ((_sel_buf[PVSelection::line_index_to_chunk(r)]) & (1UL<<(PVSelection::line_index_to_chunk_bit(r)))) {
							res = r;
							break;
						}
					}
				}
			}
			_tree->_sel_elts[b] = res;
		}
	}

private:
	mutable PVParallelView::PVZoneTree* _tree;
	Picviz::PVSelection::const_pointer _sel_buf;
};

} }

void PVParallelView::PVZoneTree::filter_by_sel_tbb_treeb_new(PVZoneProcessing const& zp, const Picviz::PVSelection& sel)
{
	BENCH_START(subtree);
	memset(_sel_elts, PVROW_INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
	const uint32_t* pcol_a = zp.get_plotted_col(zp.col_a());
	const uint32_t* pcol_b = zp.get_plotted_col(zp.col_b());

	sel.visit_selected_lines([&](PVRow r){
		const PVRow y1 = pcol_a[r] >> (32-NBITS_INDEX);
		const PVRow y2 = pcol_b[r] >> (32-NBITS_INDEX);

		const PVRow b = y1 | (y2 << NBITS_INDEX);
		_sel_elts[b] = picviz_min(_sel_elts[b], r);
	}, zp.nrows());
	BENCH_END(subtree, "filter_by_sel_tbb_treeb_new", 1, 1, sizeof(PVRow), zp.nrows());
}

// PVZoneTree implementation
//

PVParallelView::PVZoneTree::PVZoneTree():
	PVZoneTreeBase(),
	_tree_data(NULL)
{
}

void PVParallelView::PVZoneTree::process_tbb_sse_treeb(PVZoneProcessing const& zp, ProcessData& pdata)
{

	PVRow nrows = zp.nrows();

	assert(nrows <= CUSTOMER_LINESNUMBER);

	for (uint32_t task = 0 ; task < pdata.ntasks ; task++) {
		memset(pdata.first_elts[task].elems, PVROW_INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
	}

	BENCH_START(trees);
	tbb::task_group group;
	PVTreeParams create_tree_params(zp, pdata, nrows);
	for (uint32_t task_num = 0; task_num < pdata.ntasks; ++task_num) {
		group.run(__impl::TBBCreateTreeTask(create_tree_params, task_num));
	}
	group.wait();
	BENCH_END(trees, "TREES", nrows*2, sizeof(float), nrows*2, sizeof(float));

	memset(_treeb, 0, sizeof(PVBranch)*NBUCKETS);

	__impl::TBBComputeAllocSizeAndFirstElts reduce_body(this, pdata);
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, NBUCKETS, GRAINSIZE), reduce_body, tbb::simple_partitioner());

	if (_tree_data) {
		PVCore::PVAlignedAllocator<PVRow, 16>().deallocate(_tree_data, reduce_body.alloc_size());
	}
	_tree_data = PVCore::PVAlignedAllocator<PVRow, 16>().allocate(reduce_body.alloc_size());

	// Update branch pointer
	PVRow* cur_p = _tree_data;
	for (PVRow b = 0; b < NBUCKETS; b++) {
		if (_treeb[b].count > 0) {
			_treeb[b].p = cur_p;
			cur_p += ((_treeb[b].count + 15)/16)*16;
		}
	}

	// Merge trees
	BENCH_START(merge);
	PVTreeParams merge_tree_params(zp, pdata, NBUCKETS);
	for (uint32_t task_num = 0; task_num < pdata.ntasks; ++task_num) {
		group.run(__impl::TBBMergeTreesTask(this, merge_tree_params, task_num));
	}
	group.wait();
	BENCH_END(merge, "MERGE", nrows*2, sizeof(float), nrows*2, sizeof(float));
}

void PVParallelView::PVZoneTree::process_omp_sse_treeb(PVZoneProcessing const& zp)
{
	assert(zp.nrows() <= CUSTOMER_LINESNUMBER);

	const uint32_t* pcol_a = zp.get_plotted_col_a();
	const uint32_t* pcol_b = zp.get_plotted_col_b();
	tbb::tick_count start, end;

	memset(_treeb, 0, sizeof(PVBranch)*NBUCKETS);

	// Fix max number of threads
	const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();

	// Create a tree by thread
	vec_rows_t** thread_trees = new vec_rows_t*[nthreads];
	for (size_t ith=0; ith<nthreads; ith++) {
		thread_trees[ith] = new vec_rows_t[NBUCKETS];
	}

	// Create an array of first elements by thread
	PVRow** first_elts_list = new PVRow*[nthreads];
	for (size_t ith=0; ith<nthreads; ith++) {
		first_elts_list[ith] = new PVRow[NBUCKETS];
		memset(first_elts_list[ith], PVROW_INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
	}
	size_t alloc_size = 0;
#pragma omp parallel num_threads(nthreads)
	{
		// Initialize one tree per thread
		vec_rows_t* thread_tree = thread_trees[omp_get_thread_num()];

		// Initialize one first elements arrays by thread
		PVRow* first_elts = first_elts_list[omp_get_thread_num()];

		PVRow nrows_sse = (zp.nrows()/4)*4;
#pragma omp barrier
#pragma omp master
		{
			start = tbb::tick_count::now();
		}
#pragma omp for schedule(static)
		for (PVRow r = 0; r < nrows_sse; r += 4) {
			__m128i sse_y1, sse_y2, sse_bcodes;
			sse_y1 = _mm_load_si128((const __m128i*) &pcol_a[r]);
			sse_y2 = _mm_load_si128((const __m128i*) &pcol_b[r]);

			sse_y1 = _mm_srli_epi32(sse_y1, 32-NBITS_INDEX);
			sse_y2 = _mm_srli_epi32(sse_y2, 32-NBITS_INDEX);
			sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

			uint32_t b0 = _mm_extract_epi32(sse_bcodes, 0);
			if (thread_tree[b0].size() == 0 ) {
				first_elts[b0] = r+0;
			}
			thread_tree[b0].push_back(r+0);

			uint32_t b1 = _mm_extract_epi32(sse_bcodes, 1);
			if (thread_tree[b1].size() == 0 ) {
				first_elts[b1] = r+1;
			}
			thread_tree[b1].push_back(r+1);

			uint32_t b2 = _mm_extract_epi32(sse_bcodes, 2);
			if (thread_tree[b2].size() == 0 ) {
				first_elts[b2] = r+2;
			}
			thread_tree[b2].push_back(r+2);

			uint32_t b3 = _mm_extract_epi32(sse_bcodes, 3);
			if (thread_tree[b3].size() == 0 ) {
				first_elts[b3] = r+3;
			}
			thread_tree[b3].push_back(r+3);
		}
#pragma omp master
		{
			for (PVRow r = nrows_sse; r < zp.nrows(); r++) {
				uint32_t y1 = pcol_a[r];
				uint32_t y2 = pcol_b[r];

				PVBCode b;
				b.int_v = 0;
				b.s.l = y1 >> (32-NBITS_INDEX);
				b.s.r = y2 >> (32-NBITS_INDEX);

				if (thread_tree[b.int_v].size() == 0 ) {
					first_elts[b.int_v] = r;
				}
				thread_tree[b.int_v].push_back(r);
			}
		}
#pragma omp barrier
#pragma omp for reduction(+:alloc_size) schedule(dynamic, GRAINSIZE)
		// _1 Sum the number of elements contained by each branch of the final tree
		// _2 Store the first element of each branch of the final tree in a buffer
		for (PVRow b = 0; b < NBUCKETS; b++) {
			_treeb[b].count = 0;
			for (size_t ith = 0; ith < nthreads; ith++) {
				_treeb[b].count += thread_trees[ith][b].size();
				_first_elts[b] = picviz_min(_first_elts[b], first_elts_list[ith][b]);
			}
			alloc_size += (((_treeb[b].count + 15) / 16) * 16);
		}
#pragma omp barrier
#pragma omp master
		{
			_tree_data = PVCore::PVAlignedAllocator<PVRow, 16>().allocate(alloc_size);

			// Update branch pointer
			PVRow* cur_p = _tree_data;
			for (PVRow b = 0; b < NBUCKETS; b++) {
				if (_treeb[b].count > 0) {
					_treeb[b].p = cur_p;
					cur_p += ((_treeb[b].count + 15)/16)*16;
				}
			}
		}
#pragma omp barrier
#pragma omp for schedule(dynamic, GRAINSIZE)
		for (PVRow b = 0; b < NBUCKETS; b++) {
			if (_treeb[b].count == 0) {
				continue;
			}
			PVRow* cur_branch = _treeb[b].p;
			for (size_t ith = 0; ith < nthreads; ith++) {
				vec_rows_t const& c = thread_trees[ith][b];
				if (c.size() > 0) {
					memcpy(cur_branch, &c.at(0), c.size()*sizeof(PVRow));
					cur_branch += c.size();
					assert(cur_branch <= _treeb[b].p + _treeb[b].count);
				}
			}
		}

#pragma omp master
		{
			end = tbb::tick_count::now();
		}
	}

	// Cleanup
	for (size_t ith=0; ith<nthreads; ith++) {
		delete [] first_elts_list[ith];
	}
	delete [] first_elts_list;
	for (size_t ith=0; ith<nthreads; ith++) {
		delete [] thread_trees[ith];
	}
	delete [] thread_trees;

	PVLOG_INFO("OMP tree process in %0.4f ms.\n", (end-start).seconds()*1000.0);
}

void PVParallelView::PVZoneTree::filter_by_sel_omp_treeb(Picviz::PVSelection const& sel)
{
	BENCH_START(subtree);
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();
#pragma omp parallel for schedule(dynamic, GRAINSIZE) firstprivate(sel_buf) num_threads(nthreads)
	for (size_t b = 0; b < NBUCKETS; b++) {
		if (branch_valid(b)) {
			const PVRow r = get_first_elt_of_branch(b);
			bool found = false;
			if ((sel_buf[PVSelection::line_index_to_chunk(r)]) & (1UL<<(PVSelection::line_index_to_chunk_bit(r)))) {
				found = true;
			}
			else {
				for (size_t i=0; i<_treeb[b].count; i++) {
					const PVRow r = _treeb[b].p[i];
					if ((sel_buf[PVSelection::line_index_to_chunk(r)]) & (1UL<<(PVSelection::line_index_to_chunk_bit(r)))) {
						found = true;
						break;
					}
				}
			}
			if (found) {
				_sel_elts[b] = r;
			}
			else {
				_sel_elts[b] = PVROW_INVALID_VALUE;
			}
		}
	}
	//BENCH_END(subtree, "filter_by_sel_omp_treeb", _nrows*2, sizeof(float), _nrows*2, sizeof(float));
	BENCH_END(subtree, "filter_by_sel_omp_treeb", 1, 1, sizeof(PVRow), NBUCKETS);
}

void PVParallelView::PVZoneTree::filter_by_sel_tbb_treeb(Picviz::PVSelection const& sel)
{
	// returns a zone tree with only the selected lines
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	BENCH_START(subtree);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, GRAINSIZE), __impl::TBBSelFilter(this, sel_buf), tbb::simple_partitioner());
	BENCH_END(subtree, "filter_by_sel_tbb_treeb", 1, 1, sizeof(PVRow), NBUCKETS);
}

void PVParallelView::PVZoneTree::get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted, PVRow nrows, PVCol col_a, PVCol col_b)
{
	pts.reserve(NBUCKETS*4);
	for (size_t i = 0; i < NBUCKETS; i++) {
		if (branch_valid(i) > 0) {
			PVRow idx_first = get_first_elt_of_branch(i);
			pts.push_back(0.0f);
			pts.push_back(org_plotted[col_a*nrows+idx_first]);
			pts.push_back(1.0f);
			pts.push_back(org_plotted[col_b*nrows+idx_first]);
		
		}
	}
}
