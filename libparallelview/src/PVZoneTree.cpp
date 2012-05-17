#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVAlignedBlockedRange.h>
#include <pvkernel/core/PVPODTree.h>

#include <picviz/PVSelection.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVZoneTree.h>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>

#include <tbb/task_scheduler_init.h>

#include <omp.h>

namespace PVParallelView { namespace __impl {

class TBBCreateTreeNRows
{
public:
	TBBCreateTreeNRows(PVParallelView::PVZoneTree::PVTBBCreateTreeParams const& params):
		_params(params)
	{ }

	TBBCreateTreeNRows(TBBCreateTreeNRows& x, tbb::split):
		_params(x._params)
	{ }

public:

	void operator() (const PVCore::PVAlignedBlockedRange<size_t, 4>& range) const
	{
		PVParallelView::PVZoneProcessing const& zp = _params.zp();
		PVParallelView::PVZoneTree& ztree = _params.ztree();
		
		PVCol col_a = zp.col_a();
		PVCol col_b = zp.col_b();

		const uint32_t* pcol_a = zp.get_plotted_col(col_a);
		const uint32_t* pcol_b = zp.get_plotted_col(col_b);

		PVRow r = range.begin();
		PVRow nrows = range.end();
		PVRow nrows_sse = (nrows/4)*4;

		PVParallelView::PVZoneTree::TLS_List::reference tls_tree = ztree.tls_trees.local();
		tls_tree.resize(NBUCKETS);

		PVParallelView::PVZoneTree::TLS::reference tls_first_elts = ztree.tls_first_elts.local();
		tls_first_elts.resize(NBUCKETS, PVROW_INVALID_VALUE);

		__m128i sse_y1, sse_y2, sse_bcodes;

		for (; r < nrows_sse; r += 4) {
			sse_y1 = _mm_load_si128((const __m128i*) &pcol_a[r]);
			sse_y2 = _mm_load_si128((const __m128i*) &pcol_b[r]);

			sse_y1 = _mm_srli_epi32(sse_y1, 32-NBITS_INDEX);
			sse_y2 = _mm_srli_epi32(sse_y2, 32-NBITS_INDEX);
			sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

			uint32_t b0 = _mm_extract_epi32(sse_bcodes, 0);
			if (tls_tree[b0].size() == 0) {
				tls_first_elts[b0] = r+0;
			}
			tls_tree[b0].push_back(r+0);

			uint32_t b1 = _mm_extract_epi32(sse_bcodes, 1);
			if (tls_tree[b1].size() == 0) {
				tls_first_elts[b1] = r+1;
			}
			tls_tree[b1].push_back(r+1);

			uint32_t b2 = _mm_extract_epi32(sse_bcodes, 2);
			if (tls_tree[b2].size() == 0) {
				tls_first_elts[b2] = r+2;
			}
			tls_tree[b2].push_back(r+2);

			uint32_t b3 = _mm_extract_epi32(sse_bcodes, 3);
			if (tls_tree[b3].size() == 0) {
				tls_first_elts[b3] = r+3;
			}
			tls_tree[b3].push_back(r+3);
		}
		for (; r < nrows; r++) {
			uint32_t y1 = pcol_a[r];
			uint32_t y2 = pcol_b[r];
			PVParallelView::PVBCode code_b;
			code_b.int_v = 0;
			code_b.s.l = y1 >> (32-NBITS_INDEX);
			code_b.s.r = y2 >> (32-NBITS_INDEX);

			if (tls_tree[code_b.int_v].size() == 0) {
				tls_first_elts[code_b.int_v] = r;
			}
			tls_tree[code_b.int_v].push_back(r);
		}
	}

private:
	PVParallelView::PVZoneTree::PVTBBCreateTreeParams const& _params;
};

class TBBComputeAllocSizeAndFirstElts
{
public:
	TBBComputeAllocSizeAndFirstElts (
		PVParallelView::PVZoneTree* ztree
	) :
		_ztree(ztree),
		_alloc_size(0)
	{ }

	TBBComputeAllocSizeAndFirstElts(TBBComputeAllocSizeAndFirstElts& x, tbb::split) :
		_ztree(x._ztree),
		_alloc_size(0)
	{ }

public:
	void operator() (const tbb::blocked_range<size_t>& range) const
	{
		for (PVRow b = range.begin(); b != range.end(); ++b) {
			_ztree->_treeb[b].count = 0;
			for (PVParallelView::PVZoneTree::TLS_List::const_iterator thread_tree = _ztree->tls_trees.begin(); thread_tree != _ztree->tls_trees.end(); ++thread_tree) {
				_ztree->_treeb[b].count += (*thread_tree)[b].size();
			}
			for (PVParallelView::PVZoneTree::TLS::const_iterator first_elts = _ztree->tls_first_elts.begin(); first_elts != _ztree->tls_first_elts.end(); ++first_elts) {
				_ztree->_first_elts[b] = picviz_min(_ztree->_first_elts[b], (*first_elts)[b]);
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
	mutable size_t _alloc_size;
};

class TBBMergeTrees
{
public:
	TBBMergeTrees (PVParallelView::PVZoneTree* ztree) : _ztree(ztree) {}

	TBBMergeTrees(TBBMergeTrees& x, tbb::split) :  _ztree(x._ztree)  {}

public:

	void operator() (const PVCore::PVAlignedBlockedRange<size_t, 4>& range) const
	{
		for (PVRow b = range.begin(); b != range.end(); ++b) {
			if (_ztree->_treeb[b].count == 0) {
				continue;
			}
			PVRow* cur_branch = _ztree->_treeb[b].p;
			for (PVParallelView::PVZoneTree::TLS_List::const_iterator thread_tree = _ztree->tls_trees.begin(); thread_tree != _ztree->tls_trees.end(); ++thread_tree) {
				PVParallelView::PVZoneTree::vect const& c = (*thread_tree)[b];
				if (c.size() > 0) {
					memcpy(cur_branch, &c.at(0), c.size()*sizeof(PVRow));
					cur_branch += c.size();
					assert(cur_branch <= _ztree->_treeb[b].p + _ztree->_treeb[b].count);
				}
			}
		}
	}

private:
	PVParallelView::PVZoneTree* _ztree;
};

class TBBPF3
{
public:
	TBBPF3 (
		PVParallelView::PVZoneTree* tree,
		const Picviz::PVSelection::const_pointer sel_buf
	) :
		_tree(tree),
		_sel_buf(sel_buf)
	{
	}

	TBBPF3(TBBPF3& x, tbb::split) :  _tree(x._tree), _sel_buf(x._sel_buf)
	{}

public:

	void operator() (const tbb::blocked_range<size_t>& r) const
	{
		for (PVRow b = r.begin(); b != r.end(); ++b) {
			if (_tree->branch_valid(b)) {
				PVRow r = _tree->get_first_elt_of_branch(b);
				bool found = false;
				if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
					found = true;
				}
				else {
					for (size_t i=0; i< _tree->_treeb[b].count; i++) {
						PVRow r = _tree->_treeb[b].p[i];
						if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
							found = true;
							break;
						}
					}
				}
				if (found) {
					_tree->_sel_elts[b] = r;
				}
				else {
					_tree->_sel_elts[b] = PVROW_INVALID_VALUE;
				}
			}
		}
	}

private:
	mutable PVParallelView::PVZoneTree* _tree;
	Picviz::PVSelection::const_pointer _sel_buf;
};

} }

// PVZoneTree implementation
//

PVParallelView::PVZoneTree::PVZoneTree():
	PVZoneTreeBase()
{
	_treeb = new PVBranch[NBUCKETS];
}

void PVParallelView::PVZoneTree::process_tbb_sse_treeb(PVZoneProcessing const& zp)
{
	tbb::task_scheduler_init init(atol(getenv("NUM_THREADS")));
	PVRow nrows = zp.nrows();
	BENCH_START(trees);
	tbb::parallel_for(PVCore::PVAlignedBlockedRange<size_t, 4>(0, nrows, atol(getenv("GRAINSIZE"))), __impl::TBBCreateTreeNRows(PVTBBCreateTreeParams(*this, zp)), tbb::simple_partitioner());
	BENCH_END(trees, "TREES", nrows*2, sizeof(float), nrows*2, sizeof(float));

	memset(_treeb, 0, sizeof(PVBranch)*NBUCKETS);

	/*for (TLS_List::iterator thread_tree = tls_trees.begin(); thread_tree != tls_trees.end(); ++thread_tree) {
		for (PVRow b = 0; b < NBUCKETS; b++) {
			(*thread_tree)[b].clear();
		}
		thread_tree->clear();
	}
	for (TLS::iterator first_elts = tls_first_elts.begin(); first_elts != tls_first_elts.end(); ++first_elts) {
		first_elts->clear();
	}*/

	__impl::TBBComputeAllocSizeAndFirstElts reduce_body(this);
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, NBUCKETS, atol(getenv("GRAINSIZE"))), reduce_body, tbb::simple_partitioner());

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
	tbb::parallel_for(PVCore::PVAlignedBlockedRange<size_t, 4>(0, NBUCKETS, atol(getenv("GRAINSIZE"))), __impl::TBBMergeTrees(this), tbb::simple_partitioner());
	BENCH_END(merge, "MERGE", nrows*2, sizeof(float), nrows*2, sizeof(float));
}

void PVParallelView::PVZoneTree::process_omp_sse_treeb(PVZoneProcessing const& zp)
{
#if 0
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	tbb::tick_count start, end;

	_treeb = new PVBranch[NBUCKETS];
	memset(_treeb, 0, sizeof(PVBranch)*NBUCKETS);

	// Fix max number of threads
	const size_t nthreads = atol(getenv("NUM_THREADS"));
	const size_t grain_size = atol(getenv("GRAINSIZE"));

	// Create a tree by thread
	Container** thread_trees = new Container*[nthreads];
	for (size_t ith=0; ith<nthreads; ith++) {
		thread_trees[ith] = new Container[NBUCKETS];
	}

	// Create an array of first elements by thread
	PVRow** first_elts_list = new PVRow*[nthreads];
	for (size_t ith=0; ith<nthreads; ith++) {
		first_elts_list[ith] = new PVRow[NBUCKETS];
		memset(first_elts_list[ith], INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
	}
	size_t alloc_size = 0;
#pragma omp parallel num_threads(nthreads)
	{
		// Initialize one tree per thread
		Container* thread_tree = thread_trees[omp_get_thread_num()];

		// Initialize one first elements arrays by thread
		PVRow* first_elts = first_elts_list[omp_get_thread_num()];

		PVRow nrows_sse = (_nrows/4)*4;
#pragma omp barrier
#pragma omp master
		{
			start = tbb::tick_count::now();
		}
#pragma omp for/* schedule(dynamic, grain_size)*/
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
			for (PVRow r = nrows_sse; r < _nrows; r++) {
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
#pragma omp for reduction(+:alloc_size)/*schedule(dynamic, grain_size)*/
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
#pragma omp for schedule(dynamic, grain_size)
		for (PVRow b = 0; b < NBUCKETS; b++) {
			if (_treeb[b].count == 0) {
				continue;
			}
			PVRow* cur_branch = _treeb[b].p;
			for (size_t ith = 0; ith < nthreads; ith++) {
				Container const& c = thread_trees[ith][b];
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
#endif
}

void PVParallelView::PVZoneTree::filter_by_sel_omp_treeb(Picviz::PVSelection const& sel)
{
	BENCH_START(subtree);
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	const size_t nthreads = atol(getenv("NUM_THREADS"));
#pragma omp parallel for schedule(dynamic, atol(getenv("GRAINSIZE"))) firstprivate(sel_buf) num_threads(nthreads)
	for (size_t b = 0; b < NBUCKETS; b++) {
		if (branch_valid(b)) {
			PVRow r = get_first_elt_of_branch(b);
			bool found = false;
			if ((sel_buf[r>>5]) & (1U<<(r&31))) {
				found = true;
			}
			else {
				for (size_t i=0; i<_treeb[b].count; i++) {
					PVRow r = _treeb[b].p[i];
					if ((sel_buf[r>>5]) & (1U<<(r&31))) {
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
	tbb::task_scheduler_init init(atol(getenv("NUM_THREADS")));
	BENCH_START(subtree);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, atol(getenv("GRAINSIZE"))), __impl::TBBPF3(this, sel_buf), tbb::simple_partitioner());
	//BENCH_END(subtree, "filter_by_sel_tbb_treeb", _nrows*2, sizeof(float), _nrows*2, sizeof(float));
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
