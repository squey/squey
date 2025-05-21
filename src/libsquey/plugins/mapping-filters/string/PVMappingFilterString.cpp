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

#include "PVMappingFilterString.h"

#include <squey/PVScalingFilter.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvkernel/core/PVSerializeObject.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <pvcop/db/read_dict.h>

#include <numeric>

/**
 * Compute integer log2 values.
 */
static uint8_t int_log2(uint16_t v)
{
	// https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogLookup
	uint32_t shift = (v > 0xFF) << 3;
	uint8_t r = shift;
	v >>= shift;

	shift = (v > 0xF) << 2;
	r |= shift;
	v >>= shift;

	shift = (v > 0x3) << 1;
	r |= shift;
	v >>= shift;

	r |= (v >> 1);

	return r;
}

static inline uint32_t compute_str_factor(char const* buf, size_t size, bool case_sensitive = true)
{
	if (size < 1) {
		return 0;
	}

	// -------------------------------------------------------------------
	// |  a (4)  |  b (0..12)  |      c (8)        |      d (8..20)      | 32 bits
	// -------------------------------------------------------------------
	// a : log2(length)
	// b : linear splitting between 'a' according to the length of remainder of the log2 computing.
	// c : the first bytes
	// d : weak bits of the sum of the remaining bytes

	// Compute "a" and set it in the first 4 bits of factor
	uint8_t shift = 32 - 4;
	const uint8_t a = int_log2(PVCore::clamp((size_t)1, size, (size_t)(1 << ((1 << 4) - 1))));
	uint32_t factor = (a + 1) << shift; // +1 to separate 1 length strings from 0 length strings

	// Compute "b" and set it in the shortest number of bits that may contains it after "a"
	// The shortest number of bits is "a"
	shift -= a;
	size_t b = (size - (1 << a));
	factor = factor | (b << shift);

	// Set the first bytes in "c"
	shift -= 8;
	uint8_t c = buf[0];
	factor = factor | (c << shift);

	// Compute the sum of remaining bytes. Truncate it on the remaining bytes (truncate strong bits)
	// and set it in "d".
	// "d" size depend on "b" size.
	size_t max_remaining_size = std::min(size, (size_t)1 << (32 - 4 - a - 8)) - 1;

	if (max_remaining_size == 0) {
		// Nothing more to sum.
		return factor;
	}

	size_t d = 0;

	if (case_sensitive) {
		d = std::accumulate(buf + 1, buf + 1 + max_remaining_size, 0, std::plus<uint32_t>());
	} else {
		d = std::accumulate(buf + 1, buf + 1 + max_remaining_size, 0, [&](uint8_t a, uint8_t b) {
			return std::tolower(a) + std::tolower(b);
		});
	}

	size_t d_bits = shift;
	// Number of bits in a char sum depend on the number of summed values.
	uint8_t bits_in_sum = 8 + int_log2(max_remaining_size);
	shift -= std::max(shift, bits_in_sum);
	// Mask strong bits and set these values as we want maximal entropy.
	factor = factor | ((d & ((1 << d_bits) - 1)) << shift);

	return factor;
}

Squey::PVMappingFilterString::PVMappingFilterString(PVCore::PVArgumentList const& args)
    : PVMappingFilter(), _case_sensitive(false)
{
	INIT_FILTER(PVMappingFilterString, args);
}

DEFAULT_ARGS_FILTER(Squey::PVMappingFilterString)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("convert-lowercase", "Convert strings to lower case")]
	    .setValue<bool>(false);
	return args;
}

void Squey::PVMappingFilterString::set_args(PVCore::PVArgumentList const& args)
{
	Squey::PVMappingFilter::set_args(args);
	_case_sensitive = !args.at("convert-lowercase").toBool();
}

pvcop::db::array Squey::PVMappingFilterString::operator()(PVCol const col,
                                                           PVRush::PVNraw const& nraw)
{
	const pvcop::db::array& array = nraw.column(col);

	using scaling_t = Squey::PVScalingFilter::value_type;
	pvcop::db::array dest(Squey::scaling_type, array.size());
	auto& dest_array = dest.to_core_array<scaling_t>();

	auto* string_dict = nraw.column_dict(col);
	if (string_dict) {
		auto& dict = *string_dict;
		std::vector<uint32_t> ret(dict.size());

#pragma omp parallel for
		for (size_t dict_idx = 0; dict_idx < dict.size(); dict_idx++) {
			const char* c = dict.key(dict_idx);
			ret[dict_idx] = compute_str_factor(c, strlen(c));
		}

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
