/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVCORE_PVMEMORY2D_H_
#define __PVCORE_PVMEMORY2D_H_

#include <stdlib.h>
#include <QRect>

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
}

#endif // __PVCORE_PVMEMORY2D_H_
