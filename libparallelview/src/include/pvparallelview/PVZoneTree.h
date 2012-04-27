#ifndef PVZONETREE_H
#define PVZONETREE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVPODTree.h>
#include <QList>
#include <picviz/PVSelection.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCode.h>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

namespace PVParallelView {

class PVZoneTreeBase
{
protected:
	typedef std::vector<float> pts_t;
public:
	void set_trans_plotted(plotted_int_t const& plotted, PVRow nrows, PVCol ncols);
	virtual void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted) = 0;
	void display(QString const& name, Picviz::PVPlotted::plotted_table_t const& org_plotted);
	inline uint32_t get_plotted_value(PVRow r, PVCol c) const { return (*_plotted)[c*_nrows_aligned + r]; }
	inline uint32_t const* get_plotted_col(PVCol c) const { return &((*_plotted)[c*_nrows_aligned]); }
protected:
	plotted_int_t const* _plotted;
	PVCol _ncols;
	PVRow _nrows;
	PVRow _nrows_aligned;
};

template <class Container>
class PVZoneTree: public PVZoneTreeBase
{
	// Ensure that container::value_type is PVRow
	BOOST_STATIC_ASSERT((boost::is_same<typename Container::value_type, PVRow>::value));

	typedef Container list_rows_t;
public:
	PVZoneTree(PVCol col_a, PVCol col_b):
		_col_a(col_a), _col_b(col_b)
	{ }
public:
	void process();
	void process_sse();
	void process_omp_sse();
	void process_tbb_concurrent_vector();
	template <bool only_first>
	PVZoneTree<Container>* filter_by_sel(Picviz::PVSelection const& sel) const;
private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted);
private:
	list_rows_t _tree[NBUCKETS];
	PVCol _col_a;
	PVCol _col_b;
};

class PVZoneTreeNoAlloc: public PVZoneTreeBase
{
	typedef PVCore::PVPODTree<uint32_t, uint32_t, NBUCKETS> Tree;
public:
	PVZoneTreeNoAlloc(PVCol col_a, PVCol col_b):
		_col_a(col_a), _col_b(col_b)
	{ }
public:
	void process_sse();
	void process_omp_sse();
private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted);
private:
	Tree _tree;
	PVCol _col_a;
	PVCol _col_b;
};

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
void PVZoneTree<Container>::process_omp_sse()
{
	// Naive processing
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	tbb::tick_count start,end;
#pragma omp parallel
	{
		// Initialize one tree per thread
		Container* thread_tree = new Container[NBUCKETS];
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

			thread_tree[_mm_extract_epi32(sse_bcodes, 0)].push_back(r+0);
			thread_tree[_mm_extract_epi32(sse_bcodes, 1)].push_back(r+1);
			thread_tree[_mm_extract_epi32(sse_bcodes, 2)].push_back(r+2);
			thread_tree[_mm_extract_epi32(sse_bcodes, 3)].push_back(r+3);
		}
#pragma omp master
		for (PVRow r = nrows_sse; r < _nrows; r++) {
			uint32_t y1 = pcol_a[r];
			uint32_t y2 = pcol_b[r];

			PVBCode b;
			b.int_v = 0;
			b.s.l = y1 >> (32-NBITS_INDEX);
			b.s.r = y2 >> (32-NBITS_INDEX);

			thread_tree[b.int_v].push_back(r);
		}
		/*
#pragma omp critical
		{
			for (size_t b = 0; b < NBUCKETS; b++) {
				Container& cur_b(thread_tree[b]);
				Container& main_b(_tree[b]);
				//main_b.reserve(main_b.size() + cur_b.size());
				//std::copy(cur_b.begin(), cur_b.end(), main_b.end());
				main_b.insert(main_b.end(), cur_b.begin(), cur_b.end());
			}
		}*/
#pragma omp barrier
#pragma omp master
		{
			end = tbb::tick_count::now();
		}
		delete [] thread_tree;
	}

	//PVLOG_INFO("OMP tree process in %0.4f ms.\n", (end-start).seconds()*1000.0);
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
template <bool only_first>
PVZoneTree<Container>* PVZoneTree<Container>::filter_by_sel(Picviz::PVSelection const& sel) const
{
	// returns a zone tree with only the selected lines
	PVZoneTree<Container>* ret = new PVZoneTree<Container>(_col_a, _col_b);
	ret->set_trans_plotted(*_plotted, _nrows, _ncols);

	const char* str_bench = (only_first) ? "subtree-first" : "subtree";
	BENCH_START(subtree);
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
#pragma omp parallel for firstprivate(sel_buf) firstprivate(ret)
	for (size_t b = 0; b < NBUCKETS; b++) {
		list_rows_t const& src(_tree[b]);
		list_rows_t& dst(ret->_tree[b]);

		typename list_rows_t::const_iterator it_src;
		for (it_src = src.begin(); it_src != src.end(); it_src++) {
			PVRow r = *it_src;
			if ((sel_buf[r>>5]) & (1U<<(r&31))) {
				dst.push_back(r);
				if (only_first) {
					break;
				}
			}
		}
	}
	BENCH_END(subtree, str_bench, _nrows*2, sizeof(float), _nrows*2, sizeof(float));

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

}

#endif
