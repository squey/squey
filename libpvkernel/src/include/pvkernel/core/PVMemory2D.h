/**
 * \file PVMemory2D.h
 *
 * Copyright (C) Picviz Labs 2013
 */


#ifndef __PVCORE_PVMEMORY2D_H_
#define __PVCORE_PVMEMORY2D_H_

#include <stdlib.h>
#include <QRect>

namespace PVCore {

	void memmove2d (
		void* source,
		size_t width,
		size_t height,
		ssize_t x_offset,
		ssize_t y_offset
	);

	void memset2d (
		void* source,
		char value,
		size_t image_width,
		size_t image_height,
		size_t rect_x,
		size_t rect_y,
		size_t rect_width,
		size_t rect_height
	);

	void memset2d (
		void* source,
		char value,
		size_t image_width,
		size_t image_height,
		const QRect& r
	);
}

#endif // __PVCORE_PVMEMORY2D_H_
