#ifndef PVCORE_PVALGORITHMS_H
#define PVCORE_PVALGORITHMS_H

#include <algorithm>

namespace PVCore {

namespace __impl {

template <class RandomAccessIterator, class Comp>
RandomAccessIterator stable_sort_find_block(RandomAccessIterator first, RandomAccessIterator last, Comp c)
{
	if (first == last) {
		return first;
	}
	RandomAccessIterator::value_type const& v_comp = *first;
	first++;
	while (first != last) {
		if (*first != v_comp) {
			break;
		}
	}
	return first;
}

template <class RandomAccessIterator, class Comp>
void stable_reverse_block(RandomAccessIterator b1_b, RandomAccessIterator b1_e, RandomAccessIterator b2_b, RandomAccessIterator b2_e)
{
}

}

template <class RandomAccessIterator, class Comp>
void stable_sort_reverse(RandomAccessIterator begin, RandomAccessIterator end, Comp c)
{
	typedef std::reverse_iterator<RandomAccessIterator> reverse_iterator;

	while (begin != end) {
		RandomAccessIterator left_block_end = __impl::stable_reverse_find_block(begin, end, c);
		if (left_block_end == end) {
			std::reverse(begin, end);
			return;
		}
		reverse_iterator right_block_begin = __impl::stable_sort_find_block(reverse_iterator(end), reverse_iterator(left_block_end), c);
		__impl::stable_reverse_block(begin, left_block_end, right_block_begin, end);
		begin = left_block_end;
		end = right_block_begin;
	}
}

}


#endif
