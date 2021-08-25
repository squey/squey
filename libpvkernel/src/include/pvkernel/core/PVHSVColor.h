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
	QRgb to_rgb() const;
	QRgb to_rgba() const;

	QColor toQColor() const;

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
