/**
 * \file parallel_stable_sort.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef TBB_PARALLEL_STABLE_SORT
#define TBB_PARALLEL_STABLE_SORT

#include <iterator>
#include <utility>
#include <memory>
#include <algorithm>
#include <iostream>

#include <tbb/parallel_invoke.h>
#include <tbb/task.h>

namespace tbb {

namespace internal {

enum { _sort_chunk_size = 7 };

#define _MOVE3(_Tp, _Up, _Vp) std::copy(_Tp, _Up, _Vp)
#define _MOVE(__val) (__val)
#define _MOVE_BACKWARD3(_Tp, _Up, _Vp) std::copy_backward(_Tp, _Up, _Vp)

template<typename RandomAccessIterator, typename Compare>
void unguarded_linear_insert(RandomAccessIterator last, Compare comp)
{
	typename std::iterator_traits<RandomAccessIterator>::value_type val = _MOVE(*last);
	RandomAccessIterator next = last;
	--next;
	while (comp(val, *next))
	{
		*last = _MOVE(*next);
		last = next;
		--next;
	}
	*last = _MOVE(val);
}

template<typename RandomAccessIterator, typename Compare>
void insertion_sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp)
{
	if (first == last)
		return;

	for (RandomAccessIterator i = first + 1; i != last; ++i)
	{
		if (comp(*i, *first))
		{
			typename std::iterator_traits<RandomAccessIterator>::value_type val = _MOVE(*i);
			_MOVE_BACKWARD3(first, i, i + 1);
			*first = _MOVE(val);
		}
		else
			unguarded_linear_insert(i, comp);
	}
}

template<typename RandomAccessIterator, typename Distance, typename Compare>
void chunk_insertion_sort(RandomAccessIterator first, RandomAccessIterator last, Distance chunk_size, Compare comp)
{
	while (last - first >= chunk_size)
	{
		insertion_sort(first, first + chunk_size, comp);
		first += chunk_size;
	}
	insertion_sort(first, last, comp);
}

template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
OutputIterator move_merge(InputIterator1 first1, InputIterator1 last1,
				InputIterator2 first2, InputIterator2 last2,
				OutputIterator result, Compare comp)
{
	while (first1 != last1 && first2 != last2)
	{
		if (comp(*first2, *first1))
		{
			*result = _MOVE(*first2);
			++first2;
		}
		else
		{
			*result = _MOVE(*first1);
			++first1;
		}
		++result;
	}
	return _MOVE3(first2, last2, _MOVE3(first1, last1, result));
}

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Distance, typename Compare>
void merge_sort_loop(RandomAccessIterator1 first,
		RandomAccessIterator1 last,
		RandomAccessIterator2 result,
		Distance step_size,
		Compare comp)
{
	const Distance two_step = 2 * step_size;

	while (last - first >= two_step)
	{
		result = move_merge(first, first + step_size,
				first + step_size,
				first + two_step, result, comp);
		first += two_step;
	}

	step_size = std::min(Distance(last - first), step_size);
	move_merge(first, first + step_size, first + step_size, last, result, comp);
}

template<typename RandomAccessIterator, typename Pointer, typename Compare>
void merge_sort_with_buffer(RandomAccessIterator begin,
		RandomAccessIterator end,
		Pointer buffer, Compare comp)
{
	typedef typename std::iterator_traits<RandomAccessIterator>::difference_type Distance;

	const Distance len = end - begin;
	const Pointer buffer_last = buffer + len;

	Distance step_size = _sort_chunk_size;
	chunk_insertion_sort(begin, end, step_size, comp);

	while (step_size < len)
	{
		merge_sort_loop(begin, end, buffer, step_size, comp);
		step_size *= 2;
		merge_sort_loop(buffer, buffer_last, begin, step_size, comp);
		step_size *= 2;
	}
}

template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
void move_merge_adaptive(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp)
{
	while (first1 != last1 && first2 != last2)
	{
		if (comp(*first2, *first1))
		{
			*result = _MOVE(*first2);
			++first2;
		}
		else 
		{
			*result = _MOVE(*first1);
			++first1;
		}
		++result;
	}
	if (first1 != last1)
		_MOVE3(first1, last1, result);
}

template<typename BidirectionalIterator1, typename BidirectionalIterator2, typename BidirectionalIterator3, typename Compare>
void move_merge_adaptive_backward(BidirectionalIterator1 first1,
		BidirectionalIterator1 last1,
		BidirectionalIterator2 first2,
		BidirectionalIterator2 last2,
		BidirectionalIterator3 result,
		Compare comp)
{
	if (first1 == last1)
	{
		_MOVE_BACKWARD3(first2, last2, result);
		return;
	}
	else if (first2 == last2)
		return;

	--last1;
	--last2;
	while (true)
	{
		if (comp(*last2, *last1))
		{
			*--result = _MOVE(*last1);
			if (first1 == last1)
			{
				_MOVE_BACKWARD3(first2, ++last2, result);
				return;
			}
			--last1;
		}
		else
		{
			*--result = _MOVE(*last2);
			if (first2 == last2)
				return;
			--last2;
		}
	}
}

template<typename BidirectionalIterator1, typename BidirectionalIterator2, typename Distance>
BidirectionalIterator1 rotate_adaptive(BidirectionalIterator1 first,
		BidirectionalIterator1 middle,
		BidirectionalIterator1 last,
		Distance len1, Distance len2,
		BidirectionalIterator2 buffer,
		Distance buffersize)
{
	BidirectionalIterator2 bufferend;
	if (len1 > len2 && len2 <= buffersize)
	{
		if (len2)
		{
			bufferend = _MOVE3(middle, last, buffer);
			_MOVE_BACKWARD3(first, middle, last);
			return _MOVE3(buffer, bufferend, first);
		}
		else
			return first;
	}
	else if (len1 <= buffersize)
	{
		if (len1)
		{
			bufferend = _MOVE3(first, middle, buffer);
			_MOVE3(middle, last, first);
			return _MOVE_BACKWARD3(buffer, bufferend, last);
		}
		else
			return last;
	}
	else
	{
		std::rotate(first, middle, last);
		std::advance(first, std::distance(middle, last));
		return first;
	}
}

template<typename BidirectionalIterator, typename Distance,  typename Pointer, typename Compare>
void merge_adaptive(BidirectionalIterator first,
		BidirectionalIterator middle,
		BidirectionalIterator last,
		Distance len1, Distance len2,
		Pointer buffer, Distance buffersize,
		Compare comp,
		tbb::task_group_context* context)
{    
	if (len1 <= len2 && len1 <= buffersize)
	{
		Pointer bufferend = _MOVE3(first, middle, buffer);
		move_merge_adaptive(buffer, bufferend, middle, last, first, comp);
	}    
	else if (len2 <= buffersize)
	{
		Pointer bufferend = _MOVE3(middle, last, buffer);
		move_merge_adaptive_backward(first, middle, buffer, bufferend, last, comp);
	}
	else 
	{
		BidirectionalIterator firstcut = first;
		BidirectionalIterator secondcut = middle;
		Distance len11 = 0; 
		Distance len22 = 0; 
		if (len1 > len2)
		{    
			len11 = len1 / 2; 
			std::advance(firstcut, len11);
			secondcut = std::lower_bound(middle, last, *firstcut, comp);
			len22 = std::distance(middle, secondcut);
		}    
		else 
		{    
			len22 = len2 / 2; 
			std::advance(secondcut, len22);
			firstcut = std::upper_bound(first, middle, *secondcut, comp);
			len11 = std::distance(first, firstcut);
		}    
		BidirectionalIterator newmiddle = rotate_adaptive(firstcut, middle, secondcut, len1 - len11, len22, buffer,	buffersize);
		/*tbb::parallel_invoke([=]{merge_adaptive(first, firstcut, newmiddle, len11, len22, buffer, buffersize, comp, context);},
		                     [=]{merge_adaptive(newmiddle, secondcut, last, len1 - len11, len2 - len22, buffer, buffersize, comp, context);}, *context);*/
		merge_adaptive(first, firstcut, newmiddle, len11, len22, buffer, buffersize, comp, context);
		merge_adaptive(newmiddle, secondcut, last, len1 - len11, len2 - len22, buffer, buffersize, comp, context);
	}
}   

template<typename RandomAccessIterator, typename Pointer, typename Distance, typename Compare>
void stable_sort_adaptive(RandomAccessIterator begin,
		RandomAccessIterator end,
		Pointer buffer, Distance buffer_size,
		Compare comp,
		tbb::task_group_context* context)
{
	const Distance len = (end - begin + 1) / 2;
	const RandomAccessIterator middle = begin + len;
	if (len > buffer_size)
	{
		//tbb::parallel_invoke([=]{stable_sort_adaptive(begin, middle, buffer, buffer_mid_size, comp, context);},
		//                     [=]{stable_sort_adaptive(middle, end, buffer + buffer_mid_size, buffer_mid_size, comp, context);}, *context);

		stable_sort_adaptive(begin, middle, buffer, buffer_size, comp, context);
		stable_sort_adaptive(middle, end, buffer,	buffer_size, comp, context);
	}
	else
	{
		merge_sort_with_buffer(begin, middle, buffer, comp);
		merge_sort_with_buffer(middle, end, buffer, comp);
	}
	merge_adaptive(begin, middle, end,
			Distance(middle - begin),
			Distance(end - middle),
			buffer, buffer_size,
			comp,
			context);
}

template<typename BidirectionalIterator, typename Distance, typename Compare>
void merge_without_buffer(BidirectionalIterator first,
		BidirectionalIterator middle,
		BidirectionalIterator last,
		Distance len1, Distance len2,
		Compare comp)
{
	if (len1 == 0 || len2 == 0)
		return;
	if (len1 + len2 == 2)
	{
		if (comp(*middle, *first))
			std::iter_swap(first, middle);
		return;
	}
	BidirectionalIterator firstcut = first;
	BidirectionalIterator secondcut = middle;
	Distance len11 = 0;
	Distance len22 = 0;
	if (len1 > len2)
	{
		len11 = len1 / 2;
		std::advance(firstcut, len11);
		secondcut = std::lower_bound(middle, last, *firstcut,
				comp);
		len22 = std::distance(middle, secondcut);
	}
	else
	{
		len22 = len2 / 2;
		std::advance(secondcut, len22);
		firstcut = std::upper_bound(first, middle, *secondcut,
				comp);
		len11 = std::distance(first, firstcut);
	}
	std::rotate(firstcut, middle, secondcut);
	BidirectionalIterator newmiddle = firstcut;
	std::advance(newmiddle, std::distance(middle, secondcut));
	merge_without_buffer(first, firstcut, newmiddle, len11, len22, comp);
	merge_without_buffer(newmiddle, secondcut, last, len1 - len11, len2 - len22, comp);
}

template<typename RandomAccessIterator, typename Compare>
void inplace_stable_sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp, tbb::task_group_context* context)
{
	if (last - first < 15)
	{    
		insertion_sort(first, last, comp);
		return;
	}
	RandomAccessIterator middle = first + (last - first) / 2; 
	tbb::parallel_invoke([=]{inplace_stable_sort(first, middle, comp, context);},
	                     [=]{inplace_stable_sort(middle, last, comp, context);}, *context);
	//inplace_stable_sort(first, middle, comp);
	//inplace_stable_sort(middle, last, comp);
	merge_without_buffer(first, middle, last,
			middle - first,
			last - middle,
			comp);
}

}

// stable_sort
template<typename RandomAccessIterator, typename Compare>
void parallel_stable_sort(RandomAccessIterator begin, RandomAccessIterator end, const Compare& comp, tbb::task_group_context& context)
{
	typedef typename std::iterator_traits<RandomAccessIterator>::value_type ValueType;
	typedef typename std::iterator_traits<RandomAccessIterator>::pointer PointerType;
	typedef typename std::iterator_traits<RandomAccessIterator>::difference_type DistanceType;

	std::pair<PointerType, ptrdiff_t> tmp_buf = std::get_temporary_buffer<ValueType>(0);
	PointerType ptmp = tmp_buf.first;
	if (ptmp) {
		ptrdiff_t size = tmp_buf.second;
		std::uninitialized_fill(ptmp, ptmp + size, *begin);
		internal::stable_sort_adaptive(begin, end, ptmp, DistanceType(size), comp, &context);
		std::return_temporary_buffer(ptmp);
	}
	else
		internal::inplace_stable_sort(begin, end, comp, &context);
}

}

#endif
