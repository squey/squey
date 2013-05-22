/**
 * \file PVMemory2D.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvkernel/core/PVMemory2D.h>

#include <stdlib.h>
#include <stdint.h>
#include <memory.h>

#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/picviz_bench.h>

void PVCore::memmove2d(
	void* source,
	size_t image_width,
	size_t image_height,
	ssize_t x_offset,
	ssize_t y_offset
)
{
	assert(x_offset < (ssize_t) image_width);
	assert(y_offset < (ssize_t) image_height);

	size_t dest_width = image_width - abs(x_offset);
	size_t dest_height = image_height - abs(y_offset);

	ssize_t offset = y_offset*image_width+x_offset;

	BENCH_START(memmove2d);

	size_t source_offset = -std::min((ssize_t)0, y_offset)*image_width-std::min((ssize_t)0, x_offset);
	size_t dest_offset = std::max((ssize_t)0, y_offset)*image_width+std::max((ssize_t)0, x_offset);
	char* s = &((char*) source)[source_offset];
	char* d = &((char*) source)[dest_offset];

	if (offset < 0) { // normal copy (front to back)
		for (uint32_t j = 0; j < dest_height; j++) {
			memmove(d, s, dest_width);
			d += image_width;
			s += image_width;
		}
	} else { // reversed copy (back to front)
		size_t reverse_offset = (dest_height-1)*image_width;
		d = (char*)d + (reverse_offset);
		s = (char*)s + (reverse_offset);
		for (uint32_t j = 0; j < dest_height; j++) {
			memmove(d, s, dest_width);
			d -= image_width;
			s -= image_width;
		}
	}

	BENCH_END(memmove2d, "memmove2d", dest_width*dest_height, sizeof(char), dest_width*dest_height, sizeof(char));
}

void PVCore::memset2d(
	void* source,
	char value,
	size_t image_width,
	size_t image_height,
	size_t rect_x,
	size_t rect_y,
	size_t rect_width,
	size_t rect_height
)
{
	assert(rect_x + rect_width < image_width);
	assert(rect_y + rect_height < image_height);

	char* s = (char*) source;

	uint32_t i = rect_y*image_width+rect_x;
	for (uint32_t j = 0; j < rect_height; j ++) {
		memset(&s[i], value, rect_width);
		i += image_width;
	}
}

void PVCore::memset2d(
	void* source,
	char value,
	size_t image_width,
	size_t image_height,
	const QRect& r)
{
	memset2d(source, value, image_width, image_height, r.x(), r.y(), r.width(), r.height());
}
