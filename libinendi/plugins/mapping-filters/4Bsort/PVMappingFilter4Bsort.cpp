/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "PVMappingFilter4Bsort.h"

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

pvcop::db::array Inendi::PVMappingFilter4Bsort::operator()(PVCol const col,
                                                           PVRush::PVNraw const& nraw)
{
	auto array = nraw.collection().column(col);

	pvcop::db::array dest(pvcop::db::type_uint32, array.size());
	auto& dest_array = dest.to_core_array<uint32_t>();

	auto* string_dict = nraw.collection().dict(col);
	if (string_dict) {
		auto& dict = *string_dict;
		std::vector<uint32_t> ret(dict.size());

		std::transform(dict.begin(), dict.end(), ret.begin(),
		               [&](const char* c) { return compute_str_factor(c, strlen(c)); });

		// Copy mapping value based on computation from dict.
		auto& core_array = array.to_core_array<uint32_t>();
		for (size_t row = 0; row < array.size(); row++) {
			dest_array[row] = ret[core_array[row]];
		}
	} else {
		for (size_t row = 0; row < array.size(); row++) {
			std::string str = array.at(row);
			dest_array[row] = compute_str_factor(str.c_str(), str.size());
		}
	}

	return dest;
}

IMPL_FILTER_NOPARAM(Inendi::PVMappingFilter4Bsort)
