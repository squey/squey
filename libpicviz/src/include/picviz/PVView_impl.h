/**
 * \file PVView_impl.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVVIEW_IMPL_H
#define PICVIZ_PVVIEW_IMPL_H

#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/PVUtils.h>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/scalable_allocator.h>
#include <boost/thread.hpp>

namespace Picviz { namespace __impl {

class ReallocableBuffer
{
public:
	ReallocableBuffer(size_t size)
	{
		_buf = scalable_malloc(size);
		_size = size;
	}

	~ReallocableBuffer()
	{
		//scalable_free(_buf);
	}

public:
	void* buffer() { return _buf; }
	const void* buffer() const { return _buf; }
	size_t size() const { return _size; }
	void reallocate(size_t size)
	{
		if (_size < size) {
			_buf = scalable_realloc(_buf, size);
			_size = size;
		}
	}

private:
	void* _buf;
	size_t _size;
};

class PVViewSortBuf
{
	typedef tbb::enumerable_thread_specific<ReallocableBuffer, tbb::cache_aligned_allocator<ReallocableBuffer>, tbb::ets_key_per_instance> ReallocableBufferTLS;

public:
	PVViewSortBuf():
		_tmp_buf(tbb::tag_tls_construct_args(), 32)
	{ }

protected:
	mutable ReallocableBufferTLS _tmp_buf;
};

template <class Tint>
struct PVViewSortAsc: public PVViewSortBuf
{
	PVViewSortAsc(PVRush::PVNraw const* nraw_, PVCol col, Picviz::PVSortingFunc_flesser f_): 
		PVViewSortBuf(), nraw(nraw_), column(col), f(f_)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		ReallocableBuffer& tmp_buf = _tmp_buf.local();
		PVCore::PVUnicodeString const s1 = nraw->at_unistr_no_cache(idx1, column);
		const size_t size_buf = s1.size()+1;
		tmp_buf.reallocate(size_buf);
		memcpy(tmp_buf.buffer(), s1.buffer(), s1.size());

		PVCore::PVUnicodeString const s2 = nraw->at_unistr_no_cache(idx2, column);

		return f(PVCore::PVUnicodeString((char*) tmp_buf.buffer(), s1.size()), s2);
	}
private:
	PVRush::PVNraw const* nraw;
	PVCol column;
	Picviz::PVSortingFunc_flesser f;
	mutable ReallocableBufferTLS _tmp_buf;
};

template <class Tint>
struct PVViewSortDesc: public PVViewSortBuf
{
	PVViewSortDesc(PVRush::PVNraw const* nraw_, PVCol col, Picviz::PVSortingFunc_flesser f_): 
		PVViewSortBuf(), nraw(nraw_), column(col), f(f_)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		ReallocableBuffer& tmp_buf = _tmp_buf.local();
		PVCore::PVUnicodeString const s1 = nraw->at_unistr(idx1, column);
		const size_t size_buf = s1.size()+1;
		tmp_buf.reallocate(size_buf);
		memcpy(tmp_buf.buffer(), s1.buffer(), s1.size());

		PVCore::PVUnicodeString const s2 = nraw->at_unistr(idx2, column);

		return f(s2, PVCore::PVUnicodeString((char*) tmp_buf.buffer(), s1.size()));
	}
private:
	PVRush::PVNraw const* nraw;
	PVCol column;
	Picviz::PVSortingFunc_flesser f;
};

template <class Tint>
struct PVViewStableSortAsc: public PVViewSortBuf
{
	PVViewStableSortAsc(PVRush::PVNraw const* nraw_, PVCol col, Picviz::PVSortingFunc_f f_): 
		PVViewSortBuf(), nraw(nraw_), column(col), f(f_)
	{
	}

	~PVViewStableSortAsc()
	{
	}

	bool operator()(Tint idx1, Tint idx2) const
	{
		ReallocableBuffer& tmp_buf = _tmp_buf.local();
		PVCore::PVUnicodeString const s1 = nraw->at_unistr_no_cache(idx1, column);
		const size_t size_buf = s1.size()+1;
		tmp_buf.reallocate(size_buf);
		memcpy(tmp_buf.buffer(), s1.buffer(), s1.size());

		PVCore::PVUnicodeString const s2 = nraw->at_unistr_no_cache(idx2, column);

		int ret = f(PVCore::PVUnicodeString((char*) tmp_buf.buffer(), s1.size()), s2);
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
struct PVViewStableSortDesc: public PVViewSortBuf
{
	PVViewStableSortDesc(PVRush::PVNraw const* nraw_, PVCol col, Picviz::PVSortingFunc_f f_): 
		PVViewSortBuf(), nraw(nraw_), column(col), f(f_)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		ReallocableBuffer& tmp_buf = _tmp_buf.local();
		PVCore::PVUnicodeString const s1 = nraw->at_unistr_no_cache(idx1, column);
		const size_t size_buf = s1.size()+1;
		tmp_buf.reallocate(size_buf);
		memcpy(tmp_buf.buffer(), s1.buffer(), s1.size());

		PVCore::PVUnicodeString const s2 = nraw->at_unistr_no_cache(idx2, column);

		int ret = f(PVCore::PVUnicodeString((char*) tmp_buf.buffer(), s1.size()), s2);
		if (ret == 0) {
			return idx1 > idx2;
		}
		return ret > 0;
	}
private:
	PVRush::PVNraw const* nraw;
	PVCol column;
	Picviz::PVSortingFunc_f f;
};

template <class Tint>
struct PVViewCompEquals: public PVViewSortBuf
{
	PVViewCompEquals(PVRush::PVNraw const* nraw_, PVCol col, Picviz::PVSortingFunc_fequals f_equals): 
		nraw(nraw_), column(col), f(f_equals)
	{ }
	bool operator()(Tint idx1, Tint idx2) const
	{
		ReallocableBuffer& tmp_buf = _tmp_buf.local();
		PVCore::PVUnicodeString const s1 = nraw->at_unistr_no_cache(idx1, column);
		const size_t size_buf = s1.size()+1;
		tmp_buf.reallocate(size_buf);
		memcpy(tmp_buf.buffer(), s1.buffer(), s1.size());

		PVCore::PVUnicodeString const s2 = nraw->at_unistr_no_cache(idx2, column);

		return f(PVCore::PVUnicodeString((char*) tmp_buf.buffer(), s1.size()), s2);
	}
private:
	PVRush::PVNraw const* nraw;
	PVCol column;
	Picviz::PVSortingFunc_fequals f;
};


template <typename Tkey>
struct PVMultisetSortAsc
{
	PVMultisetSortAsc(Picviz::PVSortingFunc_f f) : _f(f) {}

	inline bool operator()(const Tkey& p1, const Tkey& p2) const
	{
		PVCore::PVUnicodeString const s1(p1.first.c_str(), p1.first.length());
		PVCore::PVUnicodeString const s2(p2.first.c_str(), p2.first.length());

		int iret = _f(s1, s2);
		return iret < 0 || (iret == 0 && p1.second > p2.second);
	}
private:
	Picviz::PVSortingFunc_f _f;
};

template <typename Tkey>
struct PVMultisetSortDesc
{
	PVMultisetSortDesc(Picviz::PVSortingFunc_f f) : _f(f) {}

	inline bool operator()(const Tkey& p1, const Tkey& p2) const
	{
		PVCore::PVUnicodeString const s1(p1.first.c_str(), p1.first.length());
		PVCore::PVUnicodeString const s2(p2.first.c_str(), p2.first.length());

		int iret = _f(s1, s2);
		return iret > 0 || (iret == 0 && p1.second < p2.second);
	}
private:
	Picviz::PVSortingFunc_f _f;
};

typedef std::pair<std::string_tbb, uint32_t> string_index_t;

typedef std::multiset<string_index_t, PVMultisetSortAsc<string_index_t>, tbb::scalable_allocator<string_index_t>> multiset_string_index_asc_t;
typedef std::multiset<string_index_t, PVMultisetSortDesc<string_index_t>, tbb::scalable_allocator<string_index_t>> multiset_string_index_desc_t;

typedef tbb::enumerable_thread_specific<multiset_string_index_asc_t, tbb::tbb_allocator<multiset_string_index_asc_t>> multiset_string_index_asc_tls_t;
typedef tbb::enumerable_thread_specific<multiset_string_index_desc_t, tbb::tbb_allocator<multiset_string_index_desc_t>> multiset_string_index_desc_tls_t;

template <typename Multiset>
class PVMultisetReduce
{
public:
	PVMultisetReduce(const std::vector<Multiset*>& multiset_pointers, tbb::task_group_context* ctxt = NULL) : _multiset_pointers(multiset_pointers), _ctxt(ctxt) {}
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
			if unlikely(index % 10000 == 0) {
				if (_ctxt && _ctxt->is_group_execution_cancelled()) {
					return;
				}
			}
			_multiset->insert(std::move(pair));
			index++;
		}
	}

	Multiset& get_reduced_multiset() { return *_multiset; }

private:
	const std::vector<Multiset*>& _multiset_pointers;
	tbb::task_group_context* _ctxt;
	Multiset* _multiset = nullptr;
	uint32_t _index = 0;
};

template <class L>
void nraw_sort_indexes_f(PVRush::PVNraw const* nraw, PVCol col, Picviz::PVSortingFunc_f f, Qt::SortOrder order, L& idxes, tbb::task_group_context* ctxt = NULL)
{
	typedef typename L::value_type Tint;
	L tmp_indexes;
	tmp_indexes.reserve(idxes.size());
	tmp_indexes.resize(idxes.size());
	bool success = false;

	tbb::tag_tls_construct_args tag_c;

	if (order == Qt::AscendingOrder) {
		PVMultisetSortAsc<string_index_t> sort(f);
		multiset_string_index_asc_tls_t multiset_tls(tag_c, sort);
		success = stable_insert_sort_indexes_f(nraw, col, multiset_tls, sort, order, tmp_indexes, ctxt);
	}
	else {
		PVMultisetSortDesc<string_index_t> sort(f);
		multiset_string_index_desc_tls_t multiset_tls(tag_c, sort);
		success = stable_insert_sort_indexes_f(nraw, col, multiset_tls, sort, order, tmp_indexes, ctxt);
	}

	if (success) {
		idxes = std::move(tmp_indexes);
	}
}

template <class L, class Comp, class TLS>
bool stable_insert_sort_indexes_f(PVRush::PVNraw const* nraw, PVCol col, TLS& multiset_tls, Comp& /*sort*/, Qt::SortOrder /*order*/, L& idxes, tbb::task_group_context* ctxt = NULL)
{
	typedef typename TLS::value_type multiset_t;

	tbb::task_group_context my_ctxt;
	if (ctxt == NULL) {
		ctxt = &my_ctxt;
	}

	nraw->visit_column_tbb(col, [&multiset_tls](size_t i, const char* buf, size_t size)
	{
		std::string_tbb s(buf, size);
		multiset_tls.local().insert(std::move(string_index_t(s, i)));
	}, ctxt);

	// Parallel reduce
	std::vector<multiset_t*> multiset_pointers;
	multiset_pointers.reserve(multiset_tls.size());
	int index = 0;
	for (multiset_t& multiset : multiset_tls) {
		multiset_pointers[index++] = &multiset;
	}
	PVMultisetReduce<multiset_t> multiset_reduce(multiset_pointers, ctxt);

	tbb::parallel_deterministic_reduce(tbb::blocked_range<uint32_t>(0, multiset_tls.size(), 1), multiset_reduce, *ctxt);

	// Extract vector of indexes
	bool cancelled = ctxt->is_group_execution_cancelled();
	if (!cancelled) {
		const multiset_t& reduced_multiset = multiset_reduce.get_reduced_multiset();
		index = 0;
		for (const string_index_t& pair : reduced_multiset) {
			idxes[index++] = pair.second;
			if (unlikely(index % 1000 == 0 && ctxt->is_group_execution_cancelled())) {
				return false;
			}
		}
	}

	return !cancelled;
}

template <class L>
void sort_indexes_f(PVRush::PVNraw const* nraw, PVCol col, Picviz::PVSortingFunc_flesser f, Qt::SortOrder order, L& idxes, tbb::task_group_context* ctxt = NULL)
{
	typedef typename L::value_type Tint;
	tbb::task_group_context my_ctxt;
	if (ctxt == NULL) {
		ctxt = &my_ctxt;
	}

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
void stable_sort_indexes_f(PVRush::PVNraw const* nraw, PVCol col, Picviz::PVSortingFunc_f f, Qt::SortOrder order, L& idxes, tbb::task_group_context* ctxt = NULL)
{
	typedef typename L::value_type Tint;

	if (order == Qt::AscendingOrder) {
		PVViewStableSortAsc<Tint> s(nraw, col, f);
		tbb::parallel_sort(idxes.begin(), idxes.end(), s, ctxt);
	}
	else {
		PVViewStableSortDesc<Tint> s(nraw, col, f);
		tbb::parallel_sort(idxes.begin(), idxes.end(), s, ctxt);
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
