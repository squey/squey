/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "PVMappingFilterHost.h"

#include <pvkernel/rush/PVNraw.h>

#include <pvkernel/core/network.h>

#include <pvcop/db/read_dict.h>

/**
 * Return a values which sort strings based on theirs 4 first chars.
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

static uint32_t compute_mapping(const char* c, size_t size)
{
	uint32_t res;
	if (PVCore::Network::ipv4_aton(c, size, res)) {
		// That goes to the first half of the space
		// Two consecutive value may have the same mapping value.
		res >>= 1;
	} else {
		// Take the first four characters
		res = compute_str_factor(c, size);
		// That goes to the other half!
		res = (res >> 1) | 0x80000000;
	}

	return res;
}

Inendi::PVMappingFilterHost::PVMappingFilterHost() : PVMappingFilter()
{
	INIT_FILTER_NOPARAM(PVMappingFilterHost);
}

/**
 * Ip values are between 0 and 2**31 while string values are between 2**31 and 2**32
 */
pvcop::db::array Inendi::PVMappingFilterHost::operator()(PVCol const col,
                                                         PVRush::PVNraw const& nraw)
{
	auto array = nraw.collection().column(col);
	pvcop::db::array dest(pvcop::db::type_uint32, array.size());
	auto& dest_array = dest.to_core_array<uint32_t>();

	// Store mapping for each dict value.
	auto* string_dict = nraw.collection().dict(col);

	if (string_dict) {
		auto& dict = *string_dict;
		std::vector<uint32_t> ret(dict.size());
		for (size_t i = 0; i < dict.size(); i++) {
			const char* c = dict.key(i);
			ret[i] = compute_mapping(c, strlen(c));
		}

		// Copy mapping value based on computation from dict.
		auto& core_array = array.to_core_array<string_index_t>();
		for (size_t row = 0; row < array.size(); row++) {
			dest_array[row] = ret[core_array[row]];
		}
	} else {
		for (size_t row = 0; row < array.size(); row++) {
			std::string str_repr = array.at(row);
			dest_array[row] = compute_mapping(str_repr.c_str(), str_repr.size());
		}
	}

	return dest;
}
