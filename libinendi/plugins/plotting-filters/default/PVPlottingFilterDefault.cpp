/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVPlottingFilterDefault.h"

uint32_t* Inendi::PVPlottingFilterDefault::operator()(pvcop::db::array const& mapped)
{
	assert(_dest);

	switch (mapped.type()) {
	case pvcop::db::type_uint32: {
		auto& mapped_array = mapped.to_core_array<uint32_t>();
		for (size_t i = 0; i < _dest_size; i++) {
			_dest[i] = mapped_array[i];
		}
		break;
	}
	case pvcop::db::type_int32: {
		auto& mapped_array = mapped.to_core_array<int32_t>();
		for (size_t i = 0; i < _dest_size; i++) {
			const int32_t v = mapped_array[i];
			// Increase value by 2**31 to have only positive values.
			if (v >= 0) {
				_dest[i] = (uint32_t)v + (1UL << 31);
			} else {
				_dest[i] = v + (1UL << 31);
			}
		}
		break;
	}
	case pvcop::db::type_float: {
		auto& mapped_array = mapped.to_core_array<float>();
		// Pretty basic for now, and not really interesting..
		// That should also be vectorized!
		// FIXME : we assume we don't have overflow...
		for (size_t i = 0; i < _dest_size; i++) {
			_dest[i] = ((uint32_t)mapped_array[i]);
		}
		break;
	}
	default:
		assert(false);
		break;
	}
	return _dest;
}

IMPL_FILTER_NOPARAM(Inendi::PVPlottingFilterDefault)
