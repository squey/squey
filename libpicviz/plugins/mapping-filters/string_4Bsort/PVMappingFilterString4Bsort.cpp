/**
 * \file PVMappingFilterString4Bsort.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterString4Bsort.h"
#include <pvkernel/core/PVTBBMaxArray.h>
#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/PVUnicodeString16.h>

#include <tbb/parallel_reduce.h>

#include <omp.h>

inline uint32_t compute_str16_factor(PVCore::PVUnicodeString16 const& str)
{
	char b1_c = 0;
	char b2_c = 0;
	char b3_c = 0;
	char b4_c = 0;

	const size_t len = str.len();
	PVCore::PVUnicodeString16::utf_char const* const buf = str.buffer();
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
	       ((uint32_t)(b2_c) << 16) | ((uint32_t)(b1_c) << 24);
}

inline uint32_t compute_str_factor(PVCore::PVUnicodeString const& str)
{
	char b1_c = 0;
	char b2_c = 0;
	char b3_c = 0;
	char b4_c = 0;

	const size_t len = str.len();
	PVCore::PVUnicodeString::utf_char const* const buf = str.buffer();
	// TODO: check for UTF8 real chars!
	if (len >= 1) {
		b1_c = buf[0];
		if (len >= 2) {
			b2_c = buf[1];
			if (len >= 3) {
				b3_c = buf[2];
				if (len >= 4) {
					b4_c = buf[3];
				}
			}
		}
	}

	return ((uint32_t)(b4_c) << 0)  | ((uint32_t)(b3_c) << 8) |
	       ((uint32_t)(b2_c) << 16) | ((uint32_t)(b1_c) << 24);
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::string_mapping::process_utf8(const char* buf, size_t size, PVMappingFilter*)
{
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = compute_str_factor(PVCore::PVUnicodeString((const PVCore::PVUnicodeString::utf_char*) buf, size));
	return ret_ds;
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::string_mapping::process_utf16(uint16_t const* buf, size_t size, PVMappingFilter*)
{
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = compute_str16_factor(PVCore::PVUnicodeString16(buf, size));
	return ret_ds;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterString4Bsort)
