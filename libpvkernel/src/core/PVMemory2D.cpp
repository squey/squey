//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVMemory2D.h>
#include <pvkernel/core/inendi_bench.h> // for BENCH_END, BENCH_START

#include <tbb/tick_count.h> // for tick_count

#include <QRect> // for QRect

#include <algorithm> // for max, min
#include <cassert>   // for assert
#include <cstdint>   // for uint32_t
#include <cstdlib>   // for abs
#include <cstring>   // for memcpy, memset

void PVCore::memcpy2d(void* dst,
                      const void* source,
                      size_t image_width,
                      size_t image_height,
                      ssize_t x_offset,
                      ssize_t y_offset)
{
	assert(x_offset < (ssize_t)image_width);
	assert(y_offset < (ssize_t)image_height);

	size_t dest_width = image_width - std::abs(x_offset);
	size_t dest_height = image_height - std::abs(y_offset);

	BENCH_START(memcpy2d);

	size_t source_offset =
	    -std::min((ssize_t)0, y_offset) * image_width - std::min((ssize_t)0, x_offset);
	size_t dest_offset =
	    std::max((ssize_t)0, y_offset) * image_width + std::max((ssize_t)0, x_offset);
	const char* s = &((const char*)source)[source_offset];
	char* d = &((char*)dst)[dest_offset];

	for (size_t j = 0; j < dest_height; j++) {
		memcpy(d, s, dest_width);
		d += image_width;
		s += image_width;
	}

	BENCH_END(memcpy2d, "memcpy2d", dest_width * dest_height, sizeof(char),
	          dest_width * dest_height, sizeof(char));
}

void PVCore::memset2d(void* source,
                      char value,
                      size_t image_width,
                      size_t image_height,
                      size_t rect_x,
                      size_t rect_y,
                      size_t rect_width,
                      size_t rect_height)
{
	assert(rect_x + rect_width < image_width);
	assert(rect_y + rect_height < image_height);

	(void)(image_height);

	char* s = (char*)source;

	BENCH_START(memset2d);

	uint32_t i = rect_y * image_width + rect_x;
	for (uint32_t j = 0; j < rect_height; j++) {
		memset(&s[i], value, rect_width);
		i += image_width;
	}

	BENCH_END(memset2d, "memset2d", rect_width * rect_height, sizeof(char),
	          rect_width * rect_height, sizeof(char));
}

void PVCore::memset2d(
    void* source, char value, size_t image_width, size_t image_height, const QRect& r)
{
	memset2d(source, value, image_width, image_height, r.x(), r.y(), r.width(), r.height());
}
