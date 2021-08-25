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

#ifndef PVPARALLELVIEW_PVZOOMCONVERTERPOWEROFTWO_H
#define PVPARALLELVIEW_PVZOOMCONVERTERPOWEROFTWO_H

#include <pvparallelview/PVZoomConverter.h>

namespace PVParallelView
{

/**
 * @class PVZoomConverterPowerOfTwo
 *
 * This class represents a PVZoomConverter where scales
 * are defined as a power of two of zooms.
 */

class PVZoomConverterPowerOfTwo : public PVZoomConverter
{
  public:
	int scale_to_zoom(const qreal value) const override { return log2(value); }

	qreal zoom_to_scale(const int value) const override { return pow(2.0, value); }
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMCONVERTERPOWEROFTWO_H
