#ifndef PVPARALLELVIEW_PVCONTAINERZONETREE_H
#define PVPARALLELVIEW_PVCONTAINERZONETREE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAlignedBlockedRange.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVPODTree.h>
#include <picviz/PVSelection.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneTreeBase.h>
#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVBCICode.h>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>

#include <omp.h>

#include <QList>

namespace PVParallelView {

template <class Container>
class PVContainerZoneTree: public PVZoneTreeBase
{
	// Ensure that container::value_type is PVRow
	BOOST_STATIC_ASSERT((boost::is_same<typename Container::value_type, PVRow>::value));
public:
	typedef std::vector<PVRow, tbb::scalable_allocator<PVRow> > vect;
	typedef tbb::enumerable_thread_specific<vect> TLS;
public:
	typedef Container list_rows_t;
public:
	PVContainerZoneTree(PVCol col_a, PVCol col_b):
		_col_a(col_a), _col_b(col_b)
	{ }

public:
	void process_serial_no_sse();
	void process_serial_sse();
	void process_omp_sse_tree();
	void process_tbb_concurrent_vector();
	void process_tbb_sse_parallelize_on_branches();

	void filter_by_sel_omp_tree(Picviz::PVSelection const& sel);
	void filter_by_sel_tbb_tree(Picviz::PVSelection const& sel);
private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted);
public:
	list_rows_t _tree[NBUCKETS];
	PVCol _col_a;
	PVCol _col_b;
};

template <class Container>
void PVContainerZoneTree<Container>::process_serial_no_sse()
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
void PVContainerZoneTree<Container>::process_serial_sse()
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
void PVContainerZoneTree<Container>::process_omp_sse_tree()
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
		memset(first_elts, PVROW_INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
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
class TBBCreateTree {
public:
	TBBCreateTree (
		PVContainerZoneTree<Container>* ztree
	) :
		_ztree(ztree)
	{
	}

	TBBCreateTree(TBBCreateTree& x, tbb::split) :  _ztree(x._ztree) {}

	void operator() (const tbb::blocked_range<size_t>& range) const {
		const uint32_t* pcol_a = _ztree->get_plotted_col(_ztree->_col_a);
		const uint32_t* pcol_b = _ztree->get_plotted_col(_ztree->_col_b);

		for (PVRow b = range.begin(); b != range.end(); ++b) { // NBUCKETS
			PVRow nrows = _ztree->_nrows;
			PVRow nrows_sse = (_ztree->_nrows/4)*4;
			PVRow r = 0;
			for (; r < nrows_sse; r += 4) { // _nrows
				__m128i sse_y1, sse_y2, sse_bcodes;
				sse_y1 = _mm_load_si128((const __m128i*) &pcol_a[r]);
				sse_y2 = _mm_load_si128((const __m128i*) &pcol_b[r]);

				sse_y1 = _mm_srli_epi32(sse_y1, 32-NBITS_INDEX);
				sse_y2 = _mm_srli_epi32(sse_y2, 32-NBITS_INDEX);
				sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

				uint32_t b0 = _mm_extract_epi32(sse_bcodes, 0);
				if (b0 == b) {
					_ztree->_first_elts[b0] = picviz_min(_ztree->_first_elts[b0], r+0);
					_ztree->_tree[b0].push_back(r+0);
				}

				uint32_t b1 = _mm_extract_epi32(sse_bcodes, 1);
				if (b1 == b) {
					_ztree->_first_elts[b1] = picviz_min(_ztree->_first_elts[b1], r+1);
					_ztree->_tree[b1].push_back(r+1);
				}

				uint32_t b2 = _mm_extract_epi32(sse_bcodes, 2);
				if (b2 == b) {
					_ztree->_first_elts[b2] = picviz_min(_ztree->_first_elts[b2], r+2);
					_ztree->_tree[b2].push_back(r+2);
				}

				uint32_t b3 = _mm_extract_epi32(sse_bcodes, 3);
				if (b3 == b) {
					_ztree->_first_elts[b3] = picviz_min(_ztree->_first_elts[b3], r+3);
					_ztree->_tree[b3].push_back(r+3);
				}
			}
			for (r = nrows_sse; r < nrows; r++) {
				uint32_t y1 = pcol_a[r];
				uint32_t y2 = pcol_b[r];
				PVBCode code_b;
				code_b.int_v = 0;
				code_b.s.l = y1 >> (32-NBITS_INDEX);
				code_b.s.r = y2 >> (32-NBITS_INDEX);

				if (code_b.int_v == b) {
					_ztree->_first_elts[code_b.int_v] = picviz_min(_ztree->_first_elts[code_b.int_v], r);
					_ztree->_tree[code_b.int_v].push_back(r);
				}
			}
		}
	}

	mutable PVContainerZoneTree<Container>* _ztree;
};


template <class Container>
void PVContainerZoneTree<Container>::process_tbb_sse_parallelize_on_branches()
{
	tbb::task_scheduler_init init(atol(getenv("NUM_THREADS")));
	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, atol(getenv("GRAINSIZE"))), TBBCreateTree<Container>(this), tbb::simple_partitioner());
}

template <class Container>
void PVContainerZoneTree<Container>::process_tbb_concurrent_vector()
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
void PVContainerZoneTree<Container>::filter_by_sel_omp_tree(Picviz::PVSelection const& sel)
{
	// returns a zone tree with only the selected lines
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
				_sel_elts[b] = r;
			}
		}
	}
	BENCH_END(subtree, "filter_by_sel_omp_tree", _nrows*2, sizeof(float), _nrows*2, sizeof(float));
}

template <class Container>
class TBBPF2 {
public:
	TBBPF2 (
		PVContainerZoneTree<Container>* tree,
		const Picviz::PVSelection::const_pointer sel_buf,
		typename PVContainerZoneTree<Container>::TLS* tls
	) :
		_tree(tree),
		_sel_buf(sel_buf),
		_tls(tls)
	{
	}

	TBBPF2(TBBPF2& x, tbb::split) :  _tree(x._tree), _sel_buf(x._sel_buf), _tls(x._tls)
	{}

	void operator() (const tbb::blocked_range<size_t>& r) const {
		for (PVRow b = r.begin(); b != r.end(); ++b) {
			if (_tree->branch_valid(b)) {
				PVRow r = _tree->get_first_elt_of_branch(b);
				bool found = false;
				if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
					found = true;
				}
				else {
					typename PVContainerZoneTree<Container>::list_rows_t const& src(_tree->_tree[b]);
					typename PVContainerZoneTree<Container>::list_rows_t::const_iterator it_src;
					for (it_src = src.begin(); it_src != src.end(); it_src++) {
						PVRow r = *it_src;
						if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
							found = true;
							break;
						}
					}
				}
				if (found) {
					_tree->_sel_elts[b] = r;
				}
			}
		}
	}

	mutable PVContainerZoneTree<Container>* _tree;
	Picviz::PVSelection::const_pointer _sel_buf;
	typename PVContainerZoneTree<Container>::TLS* _tls;
};

template <class Container>
void PVContainerZoneTree<Container>::filter_by_sel_tbb_tree(Picviz::PVSelection const& sel)
{
	// returns a zone tree with only the selected lines
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	TLS tls;
	tbb::task_scheduler_init init(atol(getenv("NUM_THREADS")));
	BENCH_START(subtree);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, atol(getenv("GRAINSIZE"))), TBBPF2<Container>(this, sel_buf, &tls), tbb::simple_partitioner());
	BENCH_END(subtree, "filter_by_sel_tbb_tree", _nrows*2, sizeof(float), _nrows*2, sizeof(float));
}

template <class Container>
void PVContainerZoneTree<Container>::get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted)
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

}

#endif
