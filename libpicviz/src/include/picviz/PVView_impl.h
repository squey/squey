/**
 * \file PVView_impl.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVVIEW_IMPL_H
#define PICVIZ_PVVIEW_IMPL_H

#include <pvkernel/core/picviz_assert.h>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/scalable_allocator.h>
#include <boost/thread.hpp>

namespace Picviz { namespace __impl {

template <class Tint>
struct PVViewSortAsc
{
	PVViewSortAsc(PVRush::PVNraw const* nraw_, PVCol col, Picviz::PVSortingFunc_fless f_): 
		nraw(nraw_), column(col), f(f_)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		PVCore::PVUnicodeString const& s1 = nraw->at_unistr(idx1, column);
		PVCore::PVUnicodeString const& s2 = nraw->at_unistr(idx2, column);
		return f(s1, s2);
	}
private:
	PVRush::PVNraw const* nraw;
	PVCol column;
	Picviz::PVSortingFunc_fless f;
};

template <class Tint>
struct PVViewSortDesc
{
	PVViewSortDesc(PVRush::PVNraw const* nraw_, PVCol col, Picviz::PVSortingFunc_fless f_): 
		nraw(nraw_), column(col), f(f_)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		PVCore::PVUnicodeString const s1 = nraw->at_unistr(idx1, column);
		PVCore::PVUnicodeString const s2 = nraw->at_unistr(idx2, column);
		return f(s2, s1);
	}
private:
	PVRush::PVNraw const* nraw;
	PVCol column;
	Picviz::PVSortingFunc_fless f;
};

template <class Tint>
struct PVViewStableSortAsc
{
	PVViewStableSortAsc(PVRush::PVNraw const* nraw_, PVCol col, Picviz::PVSortingFunc_f f_): 
		nraw(nraw_), column(col), f(f_)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		PVCore::PVUnicodeString const& s1 = nraw->at_unistr(idx1, column);
		PVCore::PVUnicodeString const& s2 = nraw->at_unistr(idx2, column);
		int ret = f(s1, s2);
		if (ret == 0) {
			return idx1 < idx2;
		}
		return ret < 0;
	}
private:
	PVRush::PVNraw const* nraw;
	PVCol column;
	Picviz::PVSortingFunc_f f;
};

template <class Tint>
struct PVViewStableSortDesc
{
	PVViewStableSortDesc(PVRush::PVNraw const* nraw_, PVCol col, Picviz::PVSortingFunc_f f_): 
		nraw(nraw_), column(col), f(f_)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		PVCore::PVUnicodeString const s1 = nraw->at_unistr(idx1, column);
		PVCore::PVUnicodeString const s2 = nraw->at_unistr(idx2, column);
		int ret = f(s1, s2);
		if (ret == 0) {
			return idx2 > idx1;
		}
		return ret > 0;
	}
private:
	PVRush::PVNraw const* nraw;
	PVCol column;
	Picviz::PVSortingFunc_f f;
};

template <class Tint>
struct PVViewCompEquals
{
	PVViewCompEquals(PVRush::PVNraw const* nraw_, PVCol col, Picviz::PVSortingFunc_fequals f_equals): 
		nraw(nraw_), column(col), f(f_equals)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		PVCore::PVUnicodeString const s1 = nraw->at_unistr(idx1, column);
		PVCore::PVUnicodeString const s2 = nraw->at_unistr(idx2, column);
		return f(s1, s2);
	}
private:
	PVRush::PVNraw const* nraw;
	PVCol column;
	Picviz::PVSortingFunc_fequals f;
};


template <typename Tkey>
struct PVMultisetSortAsc
{
	inline bool operator()(const Tkey& p1, const Tkey& p2) const
	{
		int iret = p1.first.compare(p2.first);
		return iret < 0 || (iret == 0 && p1.second > p2.second);
	}
};

template <class L>
void stable_sort_indexes_f(PVRush::PVNraw const* nraw, PVCol col, Picviz::PVSortingFunc_f f, Qt::SortOrder order, L& idxes)
{
	typedef typename L::value_type Tint;
	if (order == Qt::AscendingOrder) {
		PVViewStableSortAsc<Tint> s(nraw, col, f);
		tbb::parallel_sort(idxes.begin(), idxes.end(), s);
	}
	else {
		PVViewStableSortDesc<Tint> s(nraw, col, f);
		tbb::parallel_sort(idxes.begin(), idxes.end(), s);
	}
}

typedef std::pair<std::string_tbb, uint32_t> string_index_t;
typedef std::multiset<string_index_t, PVMultisetSortAsc<string_index_t>, tbb::scalable_allocator<string_index_t>> multiset_string_index_t;
typedef tbb::enumerable_thread_specific<multiset_string_index_t> multiset_string_index_tls_t;

class PVMultisetReduce
{
public:
	PVMultisetReduce(const std::vector<multiset_string_index_t*>& multiset_pointers, tbb::task_group_context* ctxt = NULL) : _multiset_pointers(multiset_pointers), _ctxt(ctxt) {}
	PVMultisetReduce(PVMultisetReduce o, tbb::split) : _multiset_pointers(o._multiset_pointers), _ctxt(o._ctxt) {}

public:
	void operator() (const tbb::blocked_range<uint32_t>& range)
	{
		PV_ASSERT_VALID(range.size() == 1);
		_index = range.begin();
		_multiset = _multiset_pointers[_index];
	}

	void join(PVMultisetReduce& rhs)
	{
		int index = 0;
		for (const string_index_t& pair : *rhs._multiset) {
			if (index % 10000 == 0) {
				if (_ctxt && _ctxt->is_group_execution_cancelled()) {
					return;
				}
			}
			_multiset->insert(std::move(pair));
			index++;

			// #define likely(x)       __builtin_expect((x),1)
			// <aguinet> #define unlikely(x)     __builtin_expect((x),0)
			// <aguinet> entouré par #ifdef _GCC_
			// <aguinet> (dans le genre, pr tester si on est sous GCC°
		}
	}

	multiset_string_index_t& get_reduced_multiset() { return *_multiset; }

private:
	const std::vector<multiset_string_index_t*>& _multiset_pointers;
	tbb::task_group_context* _ctxt;
	multiset_string_index_t* _multiset = nullptr;
	uint32_t _index = 0;
};

template <class L>
bool parallel_multiset_insert_sort(PVRush::PVNraw const* nraw, const PVSelection& sel, PVCol col, L& idxes, tbb::task_group_context* ctxt = NULL)
{
	//try {
		// Parallel insertion
		multiset_string_index_tls_t multiset_tls;
		nraw->visit_column_tbb(col, [&multiset_tls](size_t i, const char* buf, size_t size)
		{
			//if (i % 10000 == 0) {
			//	boost::this_thread::interruption_point();
			//}
			std::string_tbb s(buf, size);
			/*it =*/ multiset_tls.local().insert(/*it,*/ std::move(string_index_t(s, i)));
		}, ctxt);

		// Parallel reduce
		std::vector<multiset_string_index_t*> multiset_pointers;
		multiset_pointers.reserve(multiset_tls.size());
		int index = 0;
		for (multiset_string_index_t& multiset : multiset_tls) {
			multiset_pointers[index++] = &multiset;
		}
		PVMultisetReduce multiset_reduce(multiset_pointers, ctxt);
		if (ctxt) {
			tbb::parallel_deterministic_reduce(tbb::blocked_range<uint32_t>(0, multiset_tls.size(), 1), multiset_reduce, *ctxt);
		}
		else {
			tbb::parallel_deterministic_reduce(tbb::blocked_range<uint32_t>(0, multiset_tls.size(), 1), multiset_reduce);
		}

		// Extract vector of indexes
		bool cancelled = ctxt && ctxt->is_group_execution_cancelled();
		if (!cancelled) {
			const multiset_string_index_t& reduced_multiset = multiset_reduce.get_reduced_multiset();
			index = 0;
			for (const string_index_t& pair : reduced_multiset) {
				idxes[index++] = pair.second;
				if (index % 1000 == 0 && ctxt && ctxt->is_group_execution_cancelled()) {
					return false;
				}
			}
		}
	//}
	//catch (boost::thread_interrupted const& e) {
	//	PVLOG_INFO("Sort canceled.\n");
	//	throw e;
	//}

	return !cancelled;
}

template <class L>
void nraw_sort_indexes_f(PVRush::PVNraw const* nraw, const PVSelection& sel, PVCol col, Qt::SortOrder order, L& idxes, tbb::task_group_context* ctxt = NULL)
{
	typedef typename L::value_type Tint;
	L tmp_indexes;
	tmp_indexes.resize(idxes.size());
	bool succes = false;
	if (order == Qt::AscendingOrder) {
		succes= parallel_multiset_insert_sort(nraw, sel, col, tmp_indexes, ctxt);
	}
	else {

	}

	if (succes) {
		idxes = std::move(tmp_indexes);
	}
}

template <class L>
void sort_indexes_f(PVRush::PVNraw const* nraw, PVCol col, Picviz::PVSortingFunc_fless f, Qt::SortOrder order, L& idxes)
{
	typedef typename L::value_type Tint;
	if (order == Qt::AscendingOrder) {
		PVViewSortAsc<Tint> s(nraw, col, f);
		tbb::parallel_sort(idxes.begin(), idxes.end(), s);
	}
	else {
		PVViewSortDesc<Tint> s(nraw, col, f);
		tbb::parallel_sort(idxes.begin(), idxes.end(), s);
	}
}

template <class L>
void unique_indexes_copy_f(PVRush::PVNraw const* nraw, PVCol col, Picviz::PVSortingFunc_fequals f_equals, L const& idxes_in, L& idxes_out)
{
	PVViewCompEquals<typename L::value_type> e(nraw, col, f_equals);
	std::unique_copy(idxes_in.begin(), idxes_in.end(), idxes_out.begin(), e);
}

template <class L>
typename L::iterator unique_indexes_f(PVRush::PVNraw const* nraw, PVCol col, Picviz::PVSortingFunc_fequals f_equals, L& idxes_in)
{
	PVViewCompEquals<typename L::value_type> e(nraw, col, f_equals);
	return std::unique(idxes_in.begin(), idxes_in.end(), e);
}

}

}

#endif
