/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVCORE_SERIALIZE_NUMBERS_H__
#define __PVCORE_SERIALIZE_NUMBERS_H__

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <iterator>
#include <cstddef>
#include <algorithm>

#define DECIMAL_DIGIT_ACCUMULATE(Accum, Digit_val, Type)                                           \
	((void)((void*)&(Accum) == (Type*)NULL), /* The type matches.  */ /* The type is unsigned.  */        \
	 (((Type)-1 / 10 < (Accum) || (Type)((Accum)*10 + (Digit_val)) < (Accum))                      \
	      ? false                                                                                  \
	      : (((Accum) = (Accum)*10 + (Digit_val)), true)))

namespace PVCore
{

// Taken from https://www.rosettacode.org/wiki/Range_extraction#C.2B.2B
template <typename InIter>
void serialize_numbers(InIter begin, InIter end, std::ostream& os)
{
	if (begin == end)
		return;

	int current = *begin++;
	os << current;
	int count = 1;

	while (begin != end) {
		int next = *begin++;
		if (next == current + 1)
			++count;
		else {
			if (count > 2)
				os << '-';
			else
				os << ',';
			if (count > 1)
				os << current << ',';
			os << next;
			count = 1;
		}
		current = next;
	}

	if (count > 1)
		os << (count > 2 ? '-' : ',') << current;
}

template <typename T, std::size_t n>
T* end(T (&array)[n])
{
	return array + n;
}

std::vector<std::pair<size_t, size_t>>
deserialize_numbers_as_ranges(const std::string& numbers_list);
std::vector<size_t> deserialize_numbers_as_values(const std::string& numbers_list);

size_t get_count_from_ranges(const std::vector<std::pair<size_t, size_t>>& ranges);

} // namespace PVCore

#endif // __PVCORE_SERIALIZE_NUMBERS_H__