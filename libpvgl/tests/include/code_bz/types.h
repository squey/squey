#ifndef BCODE_TYPES_H
#define BCODE_TYPES_H

#include <stdint.h>

#pragma pack(push)
#pragma pack(4)
struct PVBCode
{
	union {
		uint32_t int_v;
		struct {
			uint32_t l: 11; 
			uint32_t r: 11; 
			uint32_t type: 3;
			uint32_t __free: 7;
		} s;
	};
	void to_pts(uint16_t w, uint16_t h, uint16_t& lx, uint16_t& ly, uint16_t& rx, uint16_t& ry) const;
	void to_pts_new(uint16_t w, uint16_t h, uint16_t& lx, uint16_t& ly, uint16_t& rx, uint16_t& ry) const;
};

typedef PVBCode DECLARE_ALIGN(16) * PVBCode_ap;

#ifdef NBDEUG
#define assert_bcode_valid(b)
#else
#define assert_bcode_valid(b)\
	do {\
		assert((b).s.__free == 0);\
		assert((b).s.type <= 6);\
	} while(0);
#endif

#pragma pack(pop)

#endif
