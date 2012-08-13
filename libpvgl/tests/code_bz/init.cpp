/**
 * \file init.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <common/common.h>
#include <code_bz/types.h>
#include <code_bz/init.h>

#include <cstdlib>

void init_random_bcodes(PVBCode* ret, size_t n)
{
	PVBCode tmp;
	for (size_t i = 0; i < n; i++) {
		tmp.int_v = rand();
		tmp.s.__free = 0;
		ret[i] = tmp;
	}
}

void init_constant_bcodes(PVBCode* ret, size_t n)
{
	PVBCode tmp;
	tmp.int_v = rand();
	tmp.s.__free = 0;
	for (size_t i = 0; i < n; i++) {
		ret[i] = tmp;
	}
}

void PVBCode::to_pts(uint16_t w, uint16_t h, uint16_t& lx, uint16_t& ly, uint16_t& rx, uint16_t& ry) const
{
	lx = 0; ly = s.l;
	rx = w; ry = s.r;
}

void PVBCode::to_pts_new(uint16_t w, uint16_t h, uint16_t& lx, uint16_t& ly, uint16_t& rx, uint16_t& ry) const
{
	lx = 0; ly = s.l;
	rx = w; ry = s.r;
}
