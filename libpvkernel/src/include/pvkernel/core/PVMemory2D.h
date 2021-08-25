/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVCORE_PVMEMORY2D_H_
#define __PVCORE_PVMEMORY2D_H_

#include <cstddef>
#include <sys/types.h>
class QRect;

namespace PVCore
{

void memcpy2d(void* dst,
              const void* source,
              size_t image_width,
              size_t image_height,
              ssize_t x_offset,
              ssize_t y_offset);

void memset2d(void* source,
              char value,
              size_t image_width,
              size_t image_height,
              size_t rect_x,
              size_t rect_y,
              size_t rect_width,
              size_t rect_height);

void memset2d(void* source, char value, size_t image_width, size_t image_height, const QRect& r);
} // namespace PVCore

#endif // __PVCORE_PVMEMORY2D_H_
