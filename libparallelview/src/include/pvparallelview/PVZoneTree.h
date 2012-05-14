#ifndef PVZONETREE_H
#define PVZONETREE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVPODTree.h>
#include <QList>
#include <picviz/PVSelection.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVHSVColor.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVBCICode.h>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/enumerable_thread_specific.h"

#include <omp.h>

namespace PVParallelView {

#define INVALID_VALUE 0xFFFFFFFF
/******************************************************************************
 *
 * PVParallelView::PVZoneTreeBase
 *
 *****************************************************************************/
class PVZoneTreeBase
{
protected:
	typedef std::vector<float> pts_t;
public:
	typedef tbb::enumerable_thread_specific< std::vector<PVRow, tbb::scalable_allocator<PVRow> > > TLS;
public:
	void set_trans_plotted(plotted_int_t const& plotted, PVRow nrows, PVCol ncols);
	virtual void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted) = 0;
	void display(QString const& name, Picviz::PVPlotted::plotted_table_t const& org_plotted);
	inline uint32_t get_plotted_value(PVRow r, PVCol c) const { return (*_plotted)[c*_nrows_aligned + r]; }
	inline uint32_t const* get_plotted_col(PVCol c) const { return &((*_plotted)[c*_nrows_aligned]); }

	inline uint32_t get_first_elt_of_branch(uint32_t branch_id) const
	{
		return _first_elts[branch_id];
	}

	inline bool branch_valid(uint32_t branch_id) const
	{
		return _first_elts[branch_id] != INVALID_VALUE;
	}

	PVRow const* get_first_elts() const { return _first_elts; }

	size_t browse_tree_bci_no_sse(PVHSVColor* colors, PVBCICode* codes);
	size_t browse_tree_bci_old(PVHSVColor* colors, PVBCICode* codes);
	size_t browse_tree_bci(PVHSVColor* colors, PVBCICode* codes);

protected:
	PVRow DECLARE_ALIGN(16) _first_elts[NBUCKETS];

	plotted_int_t const* _plotted;
	PVCol _ncols;
	PVRow _nrows;
	PVRow _nrows_aligned;

};


/******************************************************************************
 *
 * PVParallelView::PVZoneTreeNoAlloc
 *
 *****************************************************************************/
class PVZoneTreeNoAlloc: public PVZoneTreeBase
{
public:
	typedef PVCore::PVPODTree<uint32_t, uint32_t, NBUCKETS> Tree;
public:
	PVZoneTreeNoAlloc(PVCol col_a, PVCol col_b):
		_col_a(col_a), _col_b(col_b)
	{ }
	/*virtual uint32_t get_first_elt_of_branch(uint32_t branch_id) const;
	virtual bool branch_valid(uint32_t branch_id) const;*/
public:
	inline const Tree* get_tree() const { return &_tree; }
	void process_sse();
	void process_omp_sse();
	PVZoneTreeNoAlloc* filter_by_sel(Picviz::PVSelection const& sel) const;
	PVZoneTreeNoAlloc* filter_by_sel_tbb(Picviz::PVSelection const& sel) const;
	size_t browse_tree_bci_by_sel(PVHSVColor* colors, PVBCICode* codes, Picviz::PVSelection const& sel);

private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted);

private:
	Tree _tree;
	PVCol _col_a;
	PVCol _col_b;
};

/******************************************************************************
 *
 * PVParallelView::PVZoneTree
 *
 *****************************************************************************/
struct PVBranch {
	PVRow* p;
	size_t count;
};

template <class Container>
class PVZoneTree: public PVZoneTreeBase
{
	// Ensure that container::value_type is PVRow
	BOOST_STATIC_ASSERT((boost::is_same<typename Container::value_type, PVRow>::value));
public:
	typedef Container list_rows_t;
public:
	PVZoneTree(PVCol col_a, PVCol col_b):
		_col_a(col_a), _col_b(col_b)
	{ }
	/*virtual uint32_t get_first_elt_of_branch(uint32_t branch_id) const;
	virtual bool branch_valid(uint32_t branch_id) const;*/

public:
	inline const list_rows_t* get_tree() const { return _tree; }
	inline PVBranch* get_treeb() const {return _treeb;}
	void process();
	void process_sse();
	void process_omp_sse_old();
	void process_omp_sse();
	void process_tbb_concurrent_vector();
	PVZoneTree<Container>* filter_by_sel(Picviz::PVSelection const& sel) const;
	PVZoneTree<Container>* filter_by_sel_tbb(Picviz::PVSelection const& sel) const;
	PVZoneTree<Container>* filter_by_sel_treeb(Picviz::PVSelection const& sel) const;
	PVZoneTree<Container>* filter_by_sel_tbb_treeb(Picviz::PVSelection const& sel) const;
private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted);
private:
	list_rows_t _tree[NBUCKETS];
	PVCol _col_a;
	PVCol _col_b;
	PVBranch* _treeb;
	PVRow* _tree_data;
};

/*template <class Container>
uint32_t PVZoneTree<Container>::get_first_elt_of_branch(uint32_t branch_id) const
{
	list_rows_t const& src(_tree[branch_id]);
	return *src.begin();
}

template <class Container>
bool PVZoneTree<Container>::branch_valid(uint32_t branch_id) const
{
	list_rows_t const& src(_tree[branch_id]);
	return src.begin() != src.end();
}*/

template <class Container>
void PVZoneTree<Container>::process()
{
	// Naive processing
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	for (PVRow r = 0; r < _nrows; r++) {
		uint32_t y1 = pcol_a[r];
		uint32_t y2 = pcol_b[r];

		PVBCode b;
		b.int_v = 0;
		b.s.l = y1 >> (32-NBITS_INDEX);
		b.s.r = y2 >> (32-NBITS_INDEX);

		_tree[b.int_v].push_back(r);
		_first_elts[b.int_v] = r;
	}
}

template <class Container>
void PVZoneTree<Container>::process_sse()
{
	// Naive processing
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	__m128i sse_y1, sse_y2, sse_bcodes;
	const PVRow nrows_sse = (_nrows/4)*4;
	for (PVRow r = 0; r < nrows_sse; r += 4) {
		sse_y1 = _mm_load_si128((const __m128i*) &pcol_a[r]);
		sse_y2 = _mm_load_si128((const __m128i*) &pcol_b[r]);

		sse_y1 = _mm_srli_epi32(sse_y1, 32-NBITS_INDEX);
		sse_y2 = _mm_srli_epi32(sse_y2, 32-NBITS_INDEX);
		sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

		uint32_t b0 = _mm_extract_epi32(sse_bcodes, 0);
		if (_tree[b0].size() == 0 ) {
			_first_elts[b0] = r+0;
		}
		_tree[b0].push_back(r+0);

		uint32_t b1 = _mm_extract_epi32(sse_bcodes, 1);
		if (_tree[b1].size() == 0 ) {
			_first_elts[b1] = r+1;
		}
		_tree[b1].push_back(r+1);

		uint32_t b2 = _mm_extract_epi32(sse_bcodes, 2);
		if (_tree[b2].size() == 0 ) {
			_first_elts[b2] = r+2;
		}
		_tree[b2].push_back(r+2);

		uint32_t b3 = _mm_extract_epi32(sse_bcodes, 3);
		if (_tree[b3].size() == 0 ) {
			_first_elts[b3] = r+3;
		}
		_tree[b3].push_back(r+3);
	}
	for (PVRow r = nrows_sse; r < _nrows; r++) {
		uint32_t y1 = pcol_a[r];
		uint32_t y2 = pcol_b[r];

		PVBCode b;
		b.int_v = 0;
		b.s.l = y1 >> (32-NBITS_INDEX);
		b.s.r = y2 >> (32-NBITS_INDEX);
		
		_tree[b.int_v].push_back(r);
		if (_tree[b.int_v].size() == 0 ) {
			_first_elts[b.int_v] = r;
		}
	}
}

template <class Container>
void PVZoneTree<Container>::process_omp_sse_old()
{
	// Naive processing
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	tbb::tick_count start,end;
	//uint32_t** thread_first_elts;
	//uint32_t* first_elts;
#pragma omp parallel num_threads(atol(getenv("NUM_THREADS")))
	{
		// Initialize one tree per thread
		Container* thread_tree = new Container[NBUCKETS];
		uint32_t* first_elts = new uint32_t[NBUCKETS];
		memset(first_elts, INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
		PVRow nrows_sse = (_nrows/4)*4;
#pragma omp barrier
#pragma omp master
		{
			start = tbb::tick_count::now();
		}
		__m128i sse_y1, sse_y2, sse_bcodes;
#pragma omp for
		for (PVRow r = 0; r < nrows_sse; r += 4) {
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

#pragma omp critical
		{
			for (size_t b = 0; b < NBUCKETS; b++) {
				Container& cur_b(thread_tree[b]);
				Container& main_b(_tree[b]);
				//main_b.reserve(main_b.size() + cur_b.size());
				//std::copy(cur_b.begin(), cur_b.end(), main_b.end());
				main_b.insert(main_b.end(), cur_b.begin(), cur_b.end());
				_first_elts[b] = picviz_min(_first_elts[b], first_elts[b]);
			}
		}
#pragma omp barrier
#pragma omp master
		{
			end = tbb::tick_count::now();
		}
		delete [] thread_tree;
		delete [] first_elts;
	}

	//PVLOG_INFO("OMP tree process in %0.4f ms.\n", (end-start).seconds()*1000.0);
}

template <class Container>
void PVZoneTree<Container>::process_omp_sse()
{
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
}

template <class Container>
void PVZoneTree<Container>::process_tbb_concurrent_vector()
{
	// Naive processing
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	__m128i sse_y1, sse_y2, sse_bcodes;
	PVRow nrows_sse = (_nrows/4)*4;
#pragma omp parallel for private(sse_y1,sse_y2,sse_bcodes) firstprivate(nrows_sse)
	for (PVRow r = 0; r < nrows_sse; r += 4) {
		sse_y1 = _mm_load_si128((const __m128i*) &pcol_a[r]);
		sse_y2 = _mm_load_si128((const __m128i*) &pcol_b[r]);

		sse_y1 = _mm_srli_epi32(sse_y1, 32-NBITS_INDEX);
		sse_y2 = _mm_srli_epi32(sse_y2, 32-NBITS_INDEX);
		sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

		_tree[_mm_extract_epi32(sse_bcodes, 0)].push_back(r+0);
		_tree[_mm_extract_epi32(sse_bcodes, 1)].push_back(r+1);
		_tree[_mm_extract_epi32(sse_bcodes, 2)].push_back(r+2);
		_tree[_mm_extract_epi32(sse_bcodes, 3)].push_back(r+3);
	}
	for (PVRow r = nrows_sse; r < _nrows; r++) {
		uint32_t y1 = pcol_a[r];
		uint32_t y2 = pcol_b[r];

		PVBCode b;
		b.int_v = 0;
		b.s.l = y1 >> (32-NBITS_INDEX);
		b.s.r = y2 >> (32-NBITS_INDEX);

		_tree[b.int_v].push_back(r);
	}
}

template <class Container>
PVZoneTree<Container>* PVZoneTree<Container>::filter_by_sel(Picviz::PVSelection const& sel) const
{
	// returns a zone tree with only the selected lines
	PVZoneTree<Container>* ret = new PVZoneTree<Container>(_col_a, _col_b);
	ret->set_trans_plotted(*_plotted, _nrows, _ncols);

	BENCH_START(subtree);
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	const size_t nthreads = atol(getenv("NUM_THREADS"));
#pragma omp parallel for schedule(dynamic, atol(getenv("GRAINSIZE"))) firstprivate(sel_buf) firstprivate(ret) num_threads(nthreads)
	for (size_t b = 0; b < NBUCKETS; b++) {
		if (branch_valid(b)) {
			list_rows_t& dst(ret->_tree[b]);
			PVRow r = get_first_elt_of_branch(b);
			bool found = false;
			if ((sel_buf[r>>5]) & (1U<<(r&31))) {
				found = true;
			}
			else {
				list_rows_t const& src(_tree[b]);
				typename list_rows_t::const_iterator it_src;
				for (it_src = src.begin(); it_src != src.end(); it_src++) {
					PVRow r = *it_src;
					if ((sel_buf[r>>5]) & (1U<<(r&31))) {
						found = true;
						break;
					}
				}
			}
			if (found) {
				dst.push_back(r);
			}
		}
	}
	BENCH_END(subtree, "filter_by_sel", _nrows*2, sizeof(float), _nrows*2, sizeof(float));

	return ret;
}

template <class Container>
PVZoneTree<Container>* PVZoneTree<Container>::filter_by_sel_treeb(Picviz::PVSelection const& sel) const
{
	// returns a zone tree with only the selected lines
	PVZoneTree<Container>* ret = new PVZoneTree<Container>(_col_a, _col_b);
	ret->set_trans_plotted(*_plotted, _nrows, _ncols);

	BENCH_START(subtree);
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	const size_t nthreads = atol(getenv("NUM_THREADS"));
#pragma omp parallel for schedule(dynamic, atol(getenv("GRAINSIZE"))) firstprivate(sel_buf) firstprivate(ret) num_threads(nthreads)
	for (size_t b = 0; b < NBUCKETS; b++) {
		if (branch_valid(b)) {
			list_rows_t& dst(ret->_tree[b]);
			PVRow r = get_first_elt_of_branch(b);
			bool found = false;
			if ((sel_buf[r>>5]) & (1U<<(r&31))) {
				found = true;
			}
			else {
				for (int i=0; i<_treeb[b].count; i++) {
					PVRow r = _treeb[b].p[i];
					if ((sel_buf[r>>5]) & (1U<<(r&31))) {
						found = true;
						break;
					}
				}
			}
			if (found) {
				dst.push_back(r);
			}
		}
	}
	BENCH_END(subtree, "filter_by_sel_treeb", _nrows*2, sizeof(float), _nrows*2, sizeof(float));

	return ret;
}

template <class Container>
class TBBPF2 {
public:
	TBBPF2 (
		const PVZoneTree<Container>* tree,
		const Picviz::PVSelection::const_pointer sel_buf,
		PVZoneTreeBase::TLS* tls
	) :
		_tree(tree),
		_sel_buf(sel_buf),
		_tls(tls)
	{
	}

	TBBPF2(TBBPF2& x, tbb::split) :  _tree(x._tree), _sel_buf(x._sel_buf), _tls(x._tls)
	{}

	void operator() (const tbb::blocked_range<size_t>& r) const {
		PVZoneTreeBase::TLS::reference tls_ref = _tls->local();
		for (PVRow b = r.begin(); b != r.end(); ++b) {
			if (_tree->branch_valid(b)) {
				PVRow r = _tree->get_first_elt_of_branch(b);
				bool found = false;
				if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
					found = true;
				}
				else {
					typename PVZoneTree<Container>::list_rows_t const& src(_tree->get_tree()[b]);
					typename PVZoneTree<Container>::list_rows_t::const_iterator it_src;
					for (it_src = src.begin(); it_src != src.end(); it_src++) {
						PVRow r = *it_src;
						if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
							found = true;
							break;
						}
					}
				}
				if (found) {
					tls_ref.push_back(r);
				}
			}
		}
	}

	const PVZoneTree<Container>* _tree;
	Picviz::PVSelection::const_pointer _sel_buf;
	PVZoneTreeBase::TLS* _tls;
};

template <class Container>
PVZoneTree<Container>* PVZoneTree<Container>::filter_by_sel_tbb(Picviz::PVSelection const& sel) const
{
	// returns a zone tree with only the selected lines
	PVZoneTree<Container>* ret = new PVZoneTree<Container>(_col_a, _col_b);
	ret->set_trans_plotted(*_plotted, _nrows, _ncols);


	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	TLS tls;
	tbb::task_scheduler_init init(atol(getenv("NUM_THREADS")));
	BENCH_START(subtree);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, atol(getenv("GRAINSIZE"))), TBBPF2<Container>(this, sel_buf, &tls), tbb::simple_partitioner());
	BENCH_END(subtree, "tbb::parallel_for", _nrows*2, sizeof(float), _nrows*2, sizeof(float));

	return ret;
}

template <class Container>
class TBBPF3 {
public:
	TBBPF3 (
		const PVZoneTree<Container>* tree,
		const Picviz::PVSelection::const_pointer sel_buf,
		PVZoneTreeBase::TLS* tls
	) :
		_tree(tree),
		_sel_buf(sel_buf),
		_tls(tls)
	{
	}

	TBBPF3(TBBPF3& x, tbb::split) :  _tree(x._tree), _sel_buf(x._sel_buf), _tls(x._tls)
	{}

	void operator() (const tbb::blocked_range<size_t>& r) const {
		PVZoneTreeBase::TLS::reference tls_ref = _tls->local();
		for (PVRow b = r.begin(); b != r.end(); ++b) {
			if (_tree->branch_valid(b)) {
				PVRow r = _tree->get_first_elt_of_branch(b);
				bool found = false;
				if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
					found = true;
				}
				else {
					for (int i=0; i< _tree->get_treeb()[b].count; i++) {
						PVRow r = _tree->get_treeb()[b].p[i];
						if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
							found = true;
							break;
						}
					}
				}
				if (found) {
					tls_ref.push_back(r);
				}
			}
		}
	}

	const PVZoneTree<Container>* _tree;
	Picviz::PVSelection::const_pointer _sel_buf;
	PVZoneTreeBase::TLS* _tls;
};

template <class Container>
PVZoneTree<Container>* PVZoneTree<Container>::filter_by_sel_tbb_treeb(Picviz::PVSelection const& sel) const
{
	// returns a zone tree with only the selected lines
	PVZoneTree<Container>* ret = new PVZoneTree<Container>(_col_a, _col_b);
	ret->set_trans_plotted(*_plotted, _nrows, _ncols);


	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	TLS tls;
	tbb::task_scheduler_init init(atol(getenv("NUM_THREADS")));
	BENCH_START(subtree);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, atol(getenv("GRAINSIZE"))), TBBPF3<Container>(this, sel_buf, &tls), tbb::simple_partitioner());
	BENCH_END(subtree, "tbb::parallel_for treeb", _nrows*2, sizeof(float), _nrows*2, sizeof(float));

	return ret;
}

template <class Container>
void PVZoneTree<Container>::get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted)
{
	pts.reserve(NBUCKETS*4);
	for (size_t i = 0; i < NBUCKETS; i++) {
		list_rows_t const& bucket(_tree[i]);
		if (bucket.size() > 0) {
			PVRow idx_first = *bucket.begin();
			pts.push_back(0.0f);
			pts.push_back(org_plotted[_col_a*_nrows+idx_first]);
			pts.push_back(1.0f);
			pts.push_back(org_plotted[_col_b*_nrows+idx_first]);
		}
	}
}

template <class Container>
class PVZoneTreeUnorderedMap: public PVZoneTreeBase
{
	// Ensure that container::value_type is PVRow
	BOOST_STATIC_ASSERT((boost::is_same<typename Container::key_type, PVRow>::value));

	typedef Container list_rows_t;
public:
	PVZoneTreeUnorderedMap(PVCol col_a, PVCol col_b):
		_col_a(col_a), _col_b(col_b), _tree(list_rows_t(NBUCKETS))
	{
	}
public:
	void process();
	void process_sse();
	void process_omp_sse();
	void process_boost();
	template <bool only_first>
	PVZoneTree<Container>* filter_by_sel(Picviz::PVSelection const& sel) const;
private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted);
private:
	list_rows_t _tree;
	PVCol _col_a;
	PVCol _col_b;
};


template <class Container>
void PVZoneTreeUnorderedMap<Container>::get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted)
{
}

template <class Container>
void PVZoneTreeUnorderedMap<Container>::process_boost()
{
	/*
	// Naive processing
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	__m128i sse_y1, sse_y2, sse_bcodes;
	const PVRow nrows_sse = (_nrows/4)*4;
	for (PVRow r = 0; r < nrows_sse; r += 4) {
		sse_y1 = _mm_load_si128((const __m128i*) &pcol_a[r]);
		sse_y2 = _mm_load_si128((const __m128i*) &pcol_b[r]);

		sse_y1 = _mm_srli_epi32(sse_y1, 32-NBITS_INDEX);
		sse_y2 = _mm_srli_epi32(sse_y2, 32-NBITS_INDEX);
		sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

		_tree[_mm_extract_epi32(sse_bcodes, 0)].push_back(r+0);
		_tree[_mm_extract_epi32(sse_bcodes, 1)].push_back(r+1);
		_tree[_mm_extract_epi32(sse_bcodes, 2)].push_back(r+2);
		_tree[_mm_extract_epi32(sse_bcodes, 3)].push_back(r+3);
	}
	for (PVRow r = nrows_sse; r < _nrows; r++) {
		uint32_t y1 = pcol_a[r];
		uint32_t y2 = pcol_b[r];

		PVBCode b;
		b.int_v = 0;
		b.s.l = y1 >> (32-NBITS_INDEX);
		b.s.r = y2 >> (32-NBITS_INDEX);

		_tree[b.int_v].push_back(r);
	}
	*/
	// Naive processing
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	for (PVRow r = 0; r < _nrows; r++) {
		uint32_t y1 = pcol_a[r];
		uint32_t y2 = pcol_b[r];

		PVBCode b;
		b.int_v = 0;
		b.s.l = y1 >> (32-NBITS_INDEX);
		b.s.r = y2 >> (32-NBITS_INDEX);

		_tree.insert(std::make_pair(b.int_v, r));
		//PVLOG_INFO("r=%d int_v=0x%x\n", r, b.int_v);
	}
}

PVZoneTreeNoAlloc* PVZoneTreeNoAlloc::filter_by_sel(Picviz::PVSelection const& sel) const
{
	// returns a zone tree with only the selected lines
	PVZoneTreeNoAlloc* ret = new PVZoneTreeNoAlloc(_col_a, _col_b);
	ret->set_trans_plotted(*_plotted, _nrows, _ncols);
	ret->_tree.resize(_nrows);

	BENCH_START(subtree);
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	const size_t nthreads = omp_get_max_threads()/2;
#pragma omp parallel for schedule(dynamic, atol(getenv("GRAINSIZE"))) firstprivate(sel_buf) firstprivate(ret) num_threads(nthreads)
	for (uint64_t b = 0; b < NBUCKETS; b++) {
		if (branch_valid(b)) {
			PVRow r = get_first_elt_of_branch(b);
			bool found = false;
			if ((sel_buf[r>>5]) & (1U<<(r&31))) {
				found = true;
			}
			else {
				Tree::const_branch_iterator it_src = _tree.begin_branch(b);
				it_src++;
				for (; it_src != _tree.end_branch(b); it_src++) {
					r = *(it_src);
					if ((sel_buf[r>>5]) & (1U<<(r&31))) {
						found = true;
						break;
					}
				}
			}
			if (found) {
				ret->_tree.push(b, r);
			}
		}
	}
	BENCH_END(subtree, "filter_by_sel", _nrows*2, sizeof(float), _nrows*2, sizeof(float));

	return ret;
}

class TBBPF {
public:
	TBBPF (
		const PVZoneTreeNoAlloc* tree,
		const Picviz::PVSelection::const_pointer sel_buf,
		PVZoneTreeBase::TLS* tls
	) :
		_tree(tree),
		_sel_buf(sel_buf),
		_tls(tls)
	{
	}

	TBBPF (TBBPF& x, tbb::split) :  _tree(x._tree), _sel_buf(x._sel_buf), _tls(x._tls)
	{}

	void operator() (const tbb::blocked_range<size_t>& r) const {
		PVZoneTreeBase::TLS::reference tls_ref = _tls->local();
		tls_ref.reserve(r.size());
		for (PVRow b = r.begin(); b != r.end(); ++b) {
			if (_tree->branch_valid(b)) {
				PVRow r = _tree->get_first_elt_of_branch(b);
				bool found = false;
				if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
					found = true;
				}
				else {
					PVZoneTreeNoAlloc::Tree::const_branch_iterator it_src = _tree->get_tree()->begin_branch(b);
					it_src++;
					for (; it_src != _tree->get_tree()->end_branch(b); it_src++) {
						r = *(it_src);
						if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
							found = true;
							break;
						}
					}
				}
				if (found) {
					tls_ref.push_back(r);
				}
			}
		}
	}

	const PVZoneTreeNoAlloc* _tree;
	Picviz::PVSelection::const_pointer _sel_buf;
	PVZoneTreeBase::TLS* _tls;
};

//#define GRAINSIZE 10000

PVZoneTreeNoAlloc* PVZoneTreeNoAlloc::filter_by_sel_tbb(Picviz::PVSelection const& sel) const
{
	// returns a zone tree with only the selected lines

	PVZoneTreeNoAlloc* ret = new PVZoneTreeNoAlloc(_col_a, _col_b);
	ret->set_trans_plotted(*_plotted, _nrows, _ncols);
	ret->_tree.resize(_nrows);


	BENCH_START(subtree);
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	TLS tls;
	tbb::task_scheduler_init init(atol(getenv("NUM_THREADS")));
	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, atol(getenv("GRAINSIZE"))), TBBPF(this, sel_buf, &tls), tbb::simple_partitioner());
	BENCH_END(subtree, "tbb::parallel_for", _nrows*2, sizeof(float), _nrows*2, sizeof(float));

	return ret;
}

}

#endif
