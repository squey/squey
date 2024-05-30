//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVMappingFilter4Bsort.h"

#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/read_dict.h>

#include <algorithm>

/**
 * Return a values which sort strings based on theirs 4 first chars.
 *
 * @warning this is a duplication from host mapping.
 */
static uint32_t compute_str_factor(const char* str, size_t len)
{
	switch (len) {
	case 0:
		return 0;
	case 1:
		return str[0] << 24;
	case 2:
		return str[0] << 24 | str[1] << 16;
	case 3:
		return str[0] << 24 | str[1] << 16 | str[2] << 8;
	default:
		return str[0] << 24 | str[1] << 16 | str[2] << 8 | str[3];
	}
}

pvcop::db::array Squey::PVMappingFilter4Bsort::operator()(PVCol const col,
                                                           PVRush::PVNraw const& nraw)
{
	const pvcop::db::array& array = nraw.column(col);

	pvcop::db::array dest("number_uint32", array.size());
	auto& dest_array = dest.to_core_array<uint32_t>();

	auto* string_dict = nraw.column_dict(col);
	if (string_dict) {
		auto& dict = *string_dict;
		std::vector<uint32_t> ret(dict.size());

#pragma omp parallel for
		for (size_t dict_idx = 0; dict_idx < dict.size(); dict_idx++) {
			const char* c = dict.key(dict_idx);
			ret[dict_idx] = compute_str_factor(c, strlen(c));
		}

		// Copy mapping value based on computation from dict.
		auto& core_array = array.to_core_array<string_index_t>();
#pragma omp parallel for
		for (size_t row = 0; row < array.size(); row++) {
			dest_array[row] = ret[core_array[row]];
		}
	} else {
#pragma omp parallel for
		for (size_t row = 0; row < array.size(); row++) {
			std::string str = array.at(row);
			dest_array[row] = compute_str_factor(str.c_str(), str.size());
		}
	}

	return dest;
}
