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

#include <pvparallelview/PVScatterViewImage.h>

#include <cassert>

#include <QImage>
#include <QPainter>

#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/PVMemory2D.h>
#include <pvkernel/core/PVLogger.h>

PVParallelView::PVScatterViewImage::PVScatterViewImage()
    : _rgb_image(image_width, image_height, QImage::Format_ARGB32)
{
	_hsv_image = new PVCore::PVHSVColor[image_width * image_height];
	clear();
}

PVParallelView::PVScatterViewImage::~PVScatterViewImage()
{
	if (_hsv_image) {
		delete[] _hsv_image;
	}
}

void PVParallelView::PVScatterViewImage::clear(const QRect& rect /* = QRect() */)
{
	if (rect.isNull()) {
		std::fill(_hsv_image, _hsv_image + (image_width * image_height), HSV_COLOR_TRANSPARENT);
	} else {
		PVCore::memset2d(_hsv_image, HSV_COLOR_TRANSPARENT.h(), image_width, image_height, rect);
	}
}

void PVParallelView::PVScatterViewImage::convert_image_from_hsv_to_rgb(QRect const& img_rect)
{
	PVCore::PVHSVColor::to_rgba(_hsv_image, _rgb_image, img_rect);
}

void PVParallelView::PVScatterViewImage::copy(PVScatterViewImage const& o)
{
	if (&o != this) {
		memcpy(_hsv_image, o._hsv_image, image_width * image_height * sizeof(PVCore::PVHSVColor));
		_rgb_image = o._rgb_image;
	}
}
