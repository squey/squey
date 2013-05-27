/**
 * \file PVHSVColor.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVHSVColor.h>

#include <assert.h>

#include <QColor>
#include <QRgb>

PVCore::PVHSVColor* PVCore::PVHSVColor::init_colors(PVRow nb_colors)
{
	PVHSVColor* colors = new PVHSVColor[nb_colors];
#pragma omp parallel for
	for (PVRow i=0; i<nb_colors; i++){
		colors[i].h() = (i/4096)%((1<<HSV_COLOR_NBITS_ZONE)*6);
	}
	return colors;
}

void PVCore::PVHSVColor::to_rgba(const PVHSVColor* hsv_image, QImage& rgb_image, QRect const& img_rect_)
{
	assert(!rbg_image.isNull());

	QRect img_rect;
	if (img_rect_.isNull()) {
		img_rect = rgb_image.rect();
	}
	else {
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

	QRgb* rgb = (QRgb*) &rgb_image.scanLine(0)[0];
/*#pragma omp parallel for schedule(static, 16)
	for (uint32_t i = rect_y*rect_width+rect_x; i < rect_width*rect_height; i++) {
		hsv_image[i].to_rgba((uint8_t*) &rgb[i]);
	}*/
#pragma omp parallel for schedule(static, 16) collapse(2)
	for (int j = rect_y; j < rect_height+rect_y; j++) {
		for (int i = rect_x; i < rect_width+rect_x; i++) {
			hsv_image[i+j*rect_width].to_rgba((uint8_t*) &rgb[j*rgb_width+i]);
		}
	}
}

static unsigned char zone2pos(unsigned char zone)
{
	const unsigned char a0 = zone&1;
	const unsigned char a1 = (zone&2)>>1;
	const unsigned char a2 = zone&4;

	return ((!(a0 ^ a1)) & (!a2)) |
		(((a1 & (!a0)) | ((a2>>2) & a0)) << 1); 
}

static unsigned char plus1mod3(unsigned char i)
{
	//return (i+1)%3;
	const unsigned char a0 = i&1;
	const unsigned char a1 = i&2;

	return (!i) | (((!a1) & a0)<<1);

}

void PVCore::PVHSVColor::to_rgb(uint8_t& r, uint8_t& g, uint8_t& b) const
{
	uint8_t rgb[3];
	to_rgb(rgb);
	r = rgb[0];
	g = rgb[1];
	b = rgb[2];
}

void PVCore::PVHSVColor::to_rgba(uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) const
{
	uint8_t rgba[4];
	to_rgba(rgba);
	r = rgba[0];
	g = rgba[1];
	b = rgba[2];
	a = rgba[3];
}

void PVCore::PVHSVColor::to_rgba(uint8_t* rgb) const
{
	if (_h == HSV_COLOR_TRANSPARENT) {
		*((uint32_t*)rgb) = 0;
	}
	else {
		to_rgb(rgb);
		rgb[3] = 0xFF;
	}
}

void PVCore::PVHSVColor::to_rgb(uint8_t* rgb) const
{
	/*uint8_t zone = _h>>HSV_COLOR_NBITS_ZONE;
	uint8_t pos = (zone%3);
	pos = pos ^ !(pos&2);
	uint8_t mask = (zone & 1)*0xFF;*/

	if (_h == HSV_COLOR_WHITE) {
		rgb[0] = 0xFF;
		rgb[1] = 0xFF;
		rgb[2] = 0xFF;
		return;
	}

	if (_h == HSV_COLOR_BLACK) {
		rgb[0] = 0;
		rgb[1] = 0;
		rgb[2] = 0;
		return;
	}

    unsigned char zone = (unsigned char) (_h>>HSV_COLOR_NBITS_ZONE);
    unsigned char pos = zone2pos(zone);
    unsigned char mask = (zone & 1)*0xFF;

	unsigned char pos2 = plus1mod3(pos);
	rgb[pos] = (uint8_t)(((_h&HSV_COLOR_MASK_ZONE)*255)>>HSV_COLOR_NBITS_ZONE) ^ mask;
	rgb[pos2] = mask;
	rgb[plus1mod3(pos2)] = 0xFF ^ mask;
}

void PVCore::PVHSVColor::toQColor(QColor& qc) const
{
	QRgb rgb;
	to_rgb((uint8_t*) &rgb);
	qc.setRgb(rgb);
}

QColor PVCore::PVHSVColor::toQColor() const
{
	QColor ret;
	toQColor(ret);
	return std::move(ret);
}

void PVCore::PVHSVColor::toQColorA(QColor& qc) const
{
	QRgb rgb;
	to_rgba((uint8_t*) &rgb);
	qc.setRgba(rgb);
}

QColor PVCore::PVHSVColor::toQColorA() const
{
	QColor ret;
	toQColorA(ret);
	return std::move(ret);
}

bool PVCore::PVHSVColor::is_valid() const
{
	// Checks that the value stored is valid
	return (_h < HSV_COLOR_COUNT) || (_h == HSV_COLOR_BLACK) || (_h == HSV_COLOR_WHITE) || (_h == HSV_COLOR_TRANSPARENT);
}
