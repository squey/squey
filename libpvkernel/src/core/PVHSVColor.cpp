/**
 * \file PVHSVColor.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVHSVColor.h>

#include <QColor>
#include <QRgb>

PVCore::PVHSVColor* PVCore::PVHSVColor::init_colors(PVRow nb_colors)
{
	PVHSVColor* colors = new PVHSVColor[nb_colors];
#pragma omp parallel for
	for (PVRow i=0; i<nb_colors; i++){
		colors[i].h() = (i)%((1<<HSV_COLOR_NBITS_ZONE)*6);
	}
	return colors;
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

void PVCore::PVHSVColor::to_rgb(uint8_t* rgb) const
{
	/*uint8_t zone = _h>>HSV_COLOR_NBITS_ZONE;
	uint8_t pos = (zone%3);
	pos = pos ^ !(pos&2);
	uint8_t mask = (zone & 1)*0xFF;*/

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
