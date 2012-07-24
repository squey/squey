/**
 * \file bcode_cb.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef BCODECB_H
#define BCODECB_H

#include <common/common.h>
#include <code_bz/types.h>
#include <stdint.h>
#include <vector>

#include <stdio.h>

typedef uint32_t DECLARE_ALIGN(16) * BCodeCB;

BCodeCB allocate_BCodeCB(void);
void free_BCodeCB(BCodeCB b);
void bcode_cb_to_bcodes(std::vector<PVBCode>& ret, BCodeCB cb);

inline uint32_t bcode2cb_idx(PVBCode c) { return c.int_v>>5; }
inline uint32_t bcode2cb_bitn(PVBCode c) { return c.int_v&31; }

inline PVBCode cb_idx2bcode(uint32_t idx, uint32_t bitn)
{
	PVBCode code;
	code.int_v = (idx<<5) + bitn;
	return code;
}

#if 0
inline uint32_t bcode2cb_idx(PVBCode c)
{
	const uint32_t l = c.s.l;
	const uint32_t r = c.s.r;
	return (l>>1) | ((r & 2046)<<9);
}
inline uint32_t bcode2cb_bitn(PVBCode c)
{
	const uint32_t l = c.s.l;
	const uint32_t r = c.s.r;
	const uint32_t lr = (l & 1) | (r & 1)<<1;
	const uint32_t bitn = lr | (c.s.type << 2);
	return bitn;
}

inline PVBCode cb_idx2bcode(uint32_t idx, uint32_t bitn)
{
	// idx: 20 significative bits
	//  10LSB = 10 last bits of L
	//  10 last = 10 last bits of R
	// bitn:
	//  first bit = first bit of L
	//  second bit = first bit of R
	//  last third bits = type
	const uint32_t l = (idx & 1023)<<1 | (bitn & 1);
	const uint32_t r = (idx & 0xffc00)>>9 | (bitn & 2);
	PVBCode code;
	code.int_v = 0;
	code.s.type = bitn>>2;
	code.s.l = l;
	code.s.r = r;
	return code;
}
#endif

#define NB_BITS_BCODE 20 // 10*2 for 'l' and 'r'
//#define NB_BCODE (1<<(NB_BITS_BCODE)) <- we can be better than that, because there is only 6 types
#define NB_BCODE ((1<<10)*(1<<10))
#define NB_INT_BCODECB (NB_BCODE/(8*sizeof(uint32_t)))
#define SIZE_BCODECB (NB_INT_BCODECB*sizeof(uint32_t))

#define TILE_SIZE_INT ((256*1024)/sizeof(int))
#define NTILE_CB ((NB_INT_BCODECB)/TILE_SIZE_INT)


#endif
