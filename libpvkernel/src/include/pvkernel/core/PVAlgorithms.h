/**
 * \file PVAlgorithms.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVALGORITHMS_H
#define PVCORE_PVALGORITHMS_H

#include <algorithm>
#include <iterator>
#include <pvkernel/core/PVFunctions.h>

namespace PVCore {

namespace __impl {

template <class RandomAccessIterator, class Comp, class Interruptible>
RandomAccessIterator stable_reverse_find_block(RandomAccessIterator first, RandomAccessIterator last, Comp c, size_t& size, Interruptible const& interrupt)
{
	if (interrupt) {
		interrupt();
	}
	size = 0;
	if (first == last) {
		return first;
	}
	typename std::iterator_traits<RandomAccessIterator>::reference v_comp = *first;
	first++;
	size++;
	while (first != last) {
		if (!c(*first, v_comp)) {
			break;
		}
		first++;
		size++;
	}
	if (interrupt) {
		interrupt();
	}
	return first;
}

template <class RandomAccessIterator, class Interruptible>
void stable_reverse_block(RandomAccessIterator b1_b, RandomAccessIterator b1_e, RandomAccessIterator b2_b, RandomAccessIterator b2_e, size_t sb1, size_t sb2, Interruptible const& interrupt)
{
	typename std::iterator_traits<RandomAccessIterator>::value_type v_tmp;
	if (sb1 == sb2) {
		while (b1_b != b1_e) {
			v_tmp = *b1_b;
			*b1_b = *b2_b;
			*b2_b = v_tmp;
			b1_b++; b2_b++;
		}
	}
	else
	if (sb1 < sb2) {
		RandomAccessIterator b1_i = b1_b;
		RandomAccessIterator b2_i = b2_e - sb1;
		RandomAccessIterator bend = b2_i;
		while (b2_i != b2_e) {
			v_tmp = *b1_i;
			*b1_i = *b2_i;
			*b2_i = v_tmp;
			b1_i++; b2_i++;
		}
		if (interrupt) {
			interrupt();
		}
		std::rotate(b1_b, b2_b, bend);
	}
	else {
		RandomAccessIterator b2_i = b2_b;
		while (b2_i != b2_e) {
			v_tmp = *b1_b;
			*b1_b = *b2_i;
			*b2_i = v_tmp;
			b1_b++; b2_i++;
		}
		if (interrupt) {
			interrupt();
		}
		std::rotate(b1_b, b1_e, b2_e);
	}
}

template <class RandomAccessIterator, class Comp, class Interruptible>
inline bool stable_sort_reverse(RandomAccessIterator begin, RandomAccessIterator end, Comp c, Interruptible const& interrupt)
{
	typedef std::reverse_iterator<RandomAccessIterator> reverse_iterator;

	size_t sb1, sb2;
	bool changed = false;
	while (begin != end) {
		RandomAccessIterator left_block_end = __impl::stable_reverse_find_block(begin, end, c, sb1, interrupt);
		if (left_block_end == end) {
			return changed;
		}
		changed = true;
		reverse_iterator right_block_begin = __impl::stable_reverse_find_block(reverse_iterator(end), reverse_iterator(left_block_end), c, sb2, interrupt);
		__impl::stable_reverse_block(begin, left_block_end, right_block_begin.base(), end, sb1, sb2, interrupt);
		begin += sb2 ;
		end -= sb1;
	}
	return changed;
}

}

template <class RandomAccessIterator, class Comp>
bool stable_sort_reverse(RandomAccessIterator begin, RandomAccessIterator end, Comp c)
{
	return __impl::stable_sort_reverse(begin, end, c, undefined_function());
}

template <class RandomAccessIterator, class Comp, class Interruptible>
bool stable_sort_reverse(RandomAccessIterator begin, RandomAccessIterator end, Comp c, Interruptible const& interrupt)
{
	return __impl::stable_sort_reverse(begin, end, c, interrupt);
}

template <typename T> T clamp(const T& value, const T& low, const T& high)
{
  return value < low ? low : (value > high ? high : value);
}

template <typename T> T min(const T& value1, const T& value2)
{
  return value1 < value2 ? value1 : value2;
}

template <typename T> T max(const T& value1, const T& value2)
{
  return value1 > value2 ? value1 : value2;
}


}


#endif
