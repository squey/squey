#ifndef PVCORE_PVALGORITHMS_H
#define PVCORE_PVALGORITHMS_H

#include <algorithm>

namespace PVCore {

namespace __impl {

template <class RandomAccessIterator, class Comp>
RandomAccessIterator stable_reverse_find_block(RandomAccessIterator first, RandomAccessIterator last, Comp c, size_t& size)
{
	size = 0;
	if (first == last) {
		return first;
	}
	typename RandomAccessIterator::value_type const& v_comp = *first;
	first++;
	size++;
	while (first != last) {
		if (!c(*first, v_comp)) {
			break;
		}
		first++;
		size++;
	}
	return first;
}

template <class RandomAccessIterator>
void stable_reverse_block(RandomAccessIterator b1_b, RandomAccessIterator b1_e, RandomAccessIterator b2_b, RandomAccessIterator b2_e, size_t sb1, size_t sb2)
{
	typename RandomAccessIterator::value_type v_tmp;
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
		std::rotate(b1_b, b1_e, b2_e);
	}
}

}

template <class RandomAccessIterator, class Comp>
void stable_sort_reverse(RandomAccessIterator begin, RandomAccessIterator end, Comp c)
{
	typedef std::reverse_iterator<RandomAccessIterator> reverse_iterator;

	size_t sb1, sb2;
	while (begin != end) {
		RandomAccessIterator left_block_end = __impl::stable_reverse_find_block(begin, end, c, sb1);
		if (left_block_end == end) {
			return;
		}
		reverse_iterator right_block_begin = __impl::stable_reverse_find_block(reverse_iterator(end), reverse_iterator(left_block_end), c, sb2);
		__impl::stable_reverse_block(begin, left_block_end, right_block_begin.base(), end, sb1, sb2);
		begin += sb2 ;
		end -= sb1;
	}
}

}


#endif
