/**
 * \file PVMappingFilterString4Bsort.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterString4Bsort.h"
#include <pvkernel/core/PVTBBMaxArray.h>

#include <tbb/parallel_reduce.h>

#include <omp.h>

inline uint32_t compute_str_factor(PVCore::PVUnicodeString const& str)
{
	char b1_c = 0;
	char b2_c = 0;
	char b3_c = 0;
	char b4_c = 0;

	const size_t len = str.len();
	PVCore::PVUnicodeString::utf_char const* const buf = str.buffer();
	if (len >= 1) {
		b1_c = QChar(buf[0]).toLatin1();
		if (len >= 2) {
			b2_c = QChar(buf[1]).toLatin1();
			if (len >= 3) {
				b3_c = QChar(buf[2]).toLatin1();
				if (len >= 4) {
					b4_c = QChar(buf[3]).toLatin1();
				}
			}
		}
	}

	return ((uint32_t)(b4_c) << 0)  | ((uint32_t)(b3_c) << 8) |
	       ((uint32_t)(b2_c) << 16) | ((uint32_t)(b4_c) << 24);
}

Picviz::PVMappingFilter::decimal_storage_type* Picviz::PVMappingFilterString4Bsort::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	// First compute all the "string factors"
	const ssize_t size = _dest_size;

#pragma omp parallel for
	for (ssize_t i = 0; i < size; i++) {
		_dest[i].storage_as_uint() = compute_str_factor(values[i]);
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterString4Bsort)
