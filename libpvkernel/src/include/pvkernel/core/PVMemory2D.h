/**
 * \file PVMemory2D.h
 *
 * Copyright (C) Picviz Labs 2013
 */


#ifndef __PVCORE_PVMEMORY2D_H_
#define __PVCORE_PVMEMORY2D_H_

#include <stdlib.h>

namespace PVCore {

void memmove2d (
	void* source,
	size_t width,
	size_t height,
	ssize_t x_offset,
	ssize_t y_offset
);

}

#endif // __PVCORE_PVMEMORY2D_H_
