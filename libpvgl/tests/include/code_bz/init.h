/**
 * \file init.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef BZCODE_INIT_H
#define BZCODE_INIT_H

#include <stdlib.h>

void init_random_bcodes(PVBCode* ret, size_t n);
void init_constant_bcodes(PVBCode* ret, size_t n);

#endif
