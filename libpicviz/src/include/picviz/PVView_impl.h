/**
 * \file PVView_impl.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVVIEW_IMPL_H
#define PICVIZ_PVVIEW_IMPL_H

#include <tbb/parallel_sort.h>

namespace Picviz { namespace __impl {

template <class Tint>
struct PVViewSortAsc
{
	PVViewSortAsc(PVRush::PVNraw* nraw_, PVCol col, Picviz::PVSortingFunc_fless f_): 
		nraw(nraw_), column(col), f(f_)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		PVCore::PVUnicodeString const& s1 = nraw->at_unistr(idx1, column);
		PVCore::PVUnicodeString const& s2 = nraw->at_unistr(idx2, column);
		return f(s1, s2);
	}
private:
	PVRush::PVNraw* nraw;
	PVCol column;
	Picviz::PVSortingFunc_fless f;
};

template <class Tint>
struct PVViewSortDesc
{
	PVViewSortDesc(PVRush::PVNraw* nraw_, PVCol col, Picviz::PVSortingFunc_fless f_): 
		nraw(nraw_), column(col), f(f_)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		PVCore::PVUnicodeString const s1 = nraw->at_unistr(idx1, column);
		PVCore::PVUnicodeString const s2 = nraw->at_unistr(idx2, column);
		return f(s2, s1);
	}
private:
	PVRush::PVNraw* nraw;
	PVCol column;
	Picviz::PVSortingFunc_fless f;
};

template <class Tint>
struct PVViewStableSortAsc
{
	PVViewStableSortAsc(PVRush::PVNraw* nraw_, PVCol col, Picviz::PVSortingFunc_f f_): 
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
	PVRush::PVNraw* nraw;
	PVCol column;
	Picviz::PVSortingFunc_f f;
};

template <class Tint>
struct PVViewStableSortDesc
{
	PVViewStableSortDesc(PVRush::PVNraw* nraw_, PVCol col, Picviz::PVSortingFunc_f f_): 
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
	PVRush::PVNraw* nraw;
	PVCol column;
	Picviz::PVSortingFunc_f f;
};

template <class Tint>
struct PVViewCompEquals
{
	PVViewCompEquals(PVRush::PVNraw* nraw_, PVCol col, Picviz::PVSortingFunc_fequals f_equals): 
		nraw(nraw_), column(col), f(f_equals)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		PVCore::PVUnicodeString const s1 = nraw->at_unistr(idx1, column);
		PVCore::PVUnicodeString const s2 = nraw->at_unistr(idx2, column);
		return f(s1, s2);
	}
private:
	PVRush::PVNraw* nraw;
	PVCol column;
	Picviz::PVSortingFunc_fequals f;
};

template <class L>
void stable_sort_indexes_f(PVRush::PVNraw* nraw, PVCol col, Picviz::PVSortingFunc_f f, Qt::SortOrder order, L& idxes)
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

template <class L>
void sort_indexes_f(PVRush::PVNraw* nraw, PVCol col, Picviz::PVSortingFunc_fless f, Qt::SortOrder order, L& idxes)
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
void unique_indexes_copy_f(PVRush::PVNraw* nraw, PVCol col, Picviz::PVSortingFunc_fequals f_equals, L const& idxes_in, L& idxes_out)
{
	PVViewCompEquals<typename L::value_type> e(nraw, col, f_equals);
	std::unique_copy(idxes_in.begin(), idxes_in.end(), idxes_out.begin(), e);
}

template <class L>
typename L::iterator unique_indexes_f(PVRush::PVNraw* nraw, PVCol col, Picviz::PVSortingFunc_fequals f_equals, L& idxes_in)
{
	PVViewCompEquals<typename L::value_type> e(nraw, col, f_equals);
	return std::unique(idxes_in.begin(), idxes_in.end(), e);
}

}

}

#endif
