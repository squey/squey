/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef BCODE_TYPES_H
#define BCODE_TYPES_H

#include <cstdint>

namespace PVParallelView
{

#pragma pack(push)
#pragma pack(4)
struct PVBCode {
	union {
		uint32_t int_v;
		struct {
			uint32_t l : 10;
			uint32_t r : 10;
			uint32_t __free : 12;
		} s;
	};
};
#pragma pack(pop)
}

#endif
