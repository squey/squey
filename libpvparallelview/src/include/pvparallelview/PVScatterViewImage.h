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

#ifndef __PVSCATTERVIEWIMAGE_H__
#define __PVSCATTERVIEWIMAGE_H__

#include <cstddef>
#include <cstdint>

#include <boost/utility.hpp>

#include <QRect>
#include <QImage>

class QPainter;

namespace PVCore
{
class PVHSVColor;
} // namespace PVCore

namespace PVParallelView
{

class PVZoomedZoneTree;

class PVScatterViewImage : boost::noncopyable
{
  public:
	constexpr static uint32_t image_width = 2048;
	constexpr static uint32_t image_height = image_width;

  public:
	PVScatterViewImage();
	PVScatterViewImage(PVScatterViewImage&& o) { swap(o); }

	~PVScatterViewImage();

  public:
	void clear(const QRect& rect = QRect());

	void convert_image_from_hsv_to_rgb(QRect const& img_rect = QRect());

	PVCore::PVHSVColor* get_hsv_image() { return _hsv_image; }
	QImage& get_rgb_image() { return _rgb_image; };

	const PVCore::PVHSVColor* get_hsv_image() const { return _hsv_image; }
	const QImage& get_rgb_image() const { return _rgb_image; };

  public:
	PVScatterViewImage& operator=(PVScatterViewImage&& o)
	{
		swap(o);
		return *this;
	}

	void swap(PVScatterViewImage& o)
	{
		if (&o != this) {
			std::swap(_hsv_image, o._hsv_image);
			_rgb_image.swap(o._rgb_image);
		}
	}

	void copy(PVScatterViewImage const& o);
	PVScatterViewImage& operator=(PVScatterViewImage const& o)
	{
		copy(o);
		return *this;
	}

  private:
	PVCore::PVHSVColor* _hsv_image;
	QImage _rgb_image;
};
} // namespace PVParallelView

#endif // __PVSCATTERVIEWIMAGE_H__
