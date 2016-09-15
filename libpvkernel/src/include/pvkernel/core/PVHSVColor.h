/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_HSVCOLOR_H
#define PVCORE_HSVCOLOR_H

#include <cstdint>

#include <QColor>
#include <QRect> // for QRect

class QImage;

constexpr const uint8_t HSV_COLOR_NBITS_ZONE = 5;
constexpr const uint8_t HSV_COLOR_MASK_ZONE = 0b00011111;

namespace PVCore
{

class PVHSVColor
{
  public:
	using h_type = uint8_t;
	static const constexpr uint8_t color_max = (1 << HSV_COLOR_NBITS_ZONE) * 6;

  public:
	PVHSVColor() : _h(0) {}
	explicit PVHSVColor(uint8_t h_) : _h(h_) {}

  public:
	inline uint8_t& h() { return _h; };
	inline uint8_t h() const { return _h; };
	static void
	to_rgba(const PVHSVColor* hsv_image, QImage& rbg_image, QRect const& img_rect = QRect());
	bool is_valid() const;

	bool operator==(PVHSVColor const& c) const { return c._h == _h; }
	bool operator!=(PVHSVColor const& c) const { return not(c == *this); }

  public:
	void to_rgb(uint8_t* rgb) const;

	void to_rgba(uint8_t* rgb) const;

	void toQColor(QColor& qc) const;
	QColor toQColor() const;

	void toQColorA(QColor& qc) const;
	QColor toQColorA() const;

  private:
	uint8_t _h;
};
} // namespace PVCore

const PVCore::PVHSVColor HSV_COLOR_WHITE(255);
const PVCore::PVHSVColor HSV_COLOR_BLACK(254);
const PVCore::PVHSVColor HSV_COLOR_TRANSPARENT(253);

const PVCore::PVHSVColor HSV_COLOR_BLUE(10);
const PVCore::PVHSVColor HSV_COLOR_GREEN(59);
const PVCore::PVHSVColor HSV_COLOR_RED(126);

#endif
