/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVHSVColor.h>

#include <cassert>

#include <QColor>
#include <QImage>
#include <QRgb>

void PVCore::PVHSVColor::to_rgba(const PVHSVColor* hsv_image,
                                 QImage& rgb_image,
                                 QRect const& img_rect_)
{
	assert(!rgb_image.isNull());

	QRect img_rect;
	if (img_rect_.isNull()) {
		img_rect = rgb_image.rect();
	} else {
		img_rect = img_rect_;
	}

	const int rect_x = img_rect.x();
	const int rect_y = img_rect.y();
	const int rect_width = img_rect.width();
	const int rect_height = img_rect.height();

	const int rgb_width = rgb_image.width();

	assert(rect_width <= rgb_width);
	assert(rect_x <= rgb_width);
	assert(rect_height <= rgb_image.height());
	assert(rect_y <= rgb_image.height());

	QRgb* rgb = (QRgb*)&rgb_image.scanLine(0)[0];
#pragma omp parallel for schedule(static, 16) collapse(2)
	for (int j = rect_y; j < rect_height + rect_y; j++) {
		for (int i = rect_x; i < rect_width + rect_x; i++) {
			rgb[j * rgb_width + i] = hsv_image[i + j * rect_width].to_rgba();
		}
	}
}

static unsigned char zone2pos(unsigned char zone)
{
	const unsigned char a0 = zone & 1;
	const unsigned char a1 = (zone & 2) >> 1;
	const unsigned char a2 = zone & 4;

	return ((!(a0 ^ a1)) & (!a2)) | (((a1 & (!a0)) | ((a2 >> 2) & a0)) << 1);
}

static unsigned char plus1mod3(unsigned char i)
{
	// return (i+1)%3;
	const unsigned char a0 = i & 1;
	const unsigned char a1 = i & 2;

	return (!i) | (((!a1) & a0) << 1);
}

QRgb PVCore::PVHSVColor::to_rgba() const
{
	if (*this == HSV_COLOR_TRANSPARENT) {
		return qRgba(0, 0, 0, 0);
	} else {
		return to_rgb();
	}
}

QRgb PVCore::PVHSVColor::to_rgb() const
{
	if (*this == HSV_COLOR_WHITE) {
		return qRgba(0xFF, 0xFF, 0xFF, 0xFF);
	}

	if (*this == HSV_COLOR_BLACK) {
		return qRgba(0, 0, 0, 0xFF);
	}

	unsigned char zone = _h >> HSV_COLOR_NBITS_ZONE;
	unsigned char pos = zone2pos(zone);
	unsigned char mask = (zone & 1) * 0xFF;

	uint8_t rgb[3];
	unsigned char pos2 = plus1mod3(pos);
	rgb[pos] = (uint8_t)(((_h & HSV_COLOR_MASK_ZONE) * 255) >> HSV_COLOR_NBITS_ZONE) ^ mask;
	rgb[pos2] = mask;
	rgb[plus1mod3(pos2)] = 0xFF ^ mask;

	return qRgba(rgb[2], rgb[1], rgb[0], 0xFF);
}

QColor PVCore::PVHSVColor::toQColor() const
{
	return QColor(to_rgb());
}

bool PVCore::PVHSVColor::is_valid() const
{
	// Checks that the value stored is valid
	return (_h < PVCore::PVHSVColor::color_max) || (*this == HSV_COLOR_BLACK) ||
	       (*this == HSV_COLOR_WHITE) || (*this == HSV_COLOR_TRANSPARENT);
}
