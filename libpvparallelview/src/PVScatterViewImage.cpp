/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVScatterViewImage.h>

#include <assert.h>

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
