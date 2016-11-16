/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterString.h"

#include <pvkernel/rush/PVNraw.h>

#include <pvkernel/core/PVSerializeObject.h>

#include <pvcop/db/read_dict.h>

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

	assert(size <= 4096UL && "PVCOP give smaller str, string size would be too big");

	// Compute "a" and set it in the first 4 bits of factor
	uint8_t shift = 32 - 4;
	const uint8_t a = int_log2(size);
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
	size_t max_remaining_size = std::min(size, 1UL << (32 - 4 - a - 8)) - 1;

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

Inendi::PVMappingFilterString::PVMappingFilterString(PVCore::PVArgumentList const& args)
    : PVMappingFilter(), _case_sensitive(false)
{
	INIT_FILTER(PVMappingFilterString, args);
}

DEFAULT_ARGS_FILTER(Inendi::PVMappingFilterString)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("convert-lowercase", "Convert strings to lower case")]
	    .setValue<bool>(false);
	return args;
}

void Inendi::PVMappingFilterString::set_args(PVCore::PVArgumentList const& args)
{
	Inendi::PVMappingFilter::set_args(args);
	_case_sensitive = !args.at("convert-lowercase").toBool();
}

pvcop::db::array Inendi::PVMappingFilterString::operator()(PVCol const col,
                                                           PVRush::PVNraw const& nraw)
{
	const pvcop::db::array& array = nraw.column(col);

	pvcop::db::array dest(pvcop::db::type_uint32, array.size());
	auto& dest_array = dest.to_core_array<uint32_t>();

	auto* string_dict = nraw.column_dict(col);
	if (string_dict) {
		auto& dict = *string_dict;
		std::vector<uint32_t> ret(dict.size());

#pragma omp parallel for
		for (size_t dict_idx = 0; dict_idx < dict.size(); dict_idx++) {
			const char* c = dict.key(dict_idx);
			ret[dict_idx] = compute_str_factor(c, strlen(c), _case_sensitive);
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
