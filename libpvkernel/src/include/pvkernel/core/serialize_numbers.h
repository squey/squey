/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
