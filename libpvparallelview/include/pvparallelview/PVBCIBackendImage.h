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

#ifndef PVPARALLELVIEW_PVBCIBACKENDIMAGE_H
#define PVPARALLELVIEW_PVBCIBACKENDIMAGE_H

#include <QImage>

namespace PVParallelView
{

/**
 * It represents an image convertible to QImage.
 */
class PVBCIBackendImage
{
  public:
	PVBCIBackendImage(uint32_t width, uint8_t height_bits)
	    : _width(width), _height_bits(height_bits)
	{
	}

	PVBCIBackendImage(PVBCIBackendImage const&) = delete;
	PVBCIBackendImage(PVBCIBackendImage&&) = delete;
	PVBCIBackendImage& operator=(PVBCIBackendImage&&) = delete;
	PVBCIBackendImage& operator=(PVBCIBackendImage const&) = delete;
	virtual ~PVBCIBackendImage() = default;

  public:
	inline QImage qimage() const { return qimage(height()); }
	virtual QImage qimage(size_t height_crop) const = 0;
	virtual bool set_width(uint32_t width)
	{
		_width = width;
		return true;
	}

	inline uint32_t width() const { return _width; }
	inline uint32_t height() const { return 1U << _height_bits; }
	inline size_t size_pixel() const { return _width * height(); }
	inline uint8_t height_bits() const { return _height_bits; }

  private:
	uint32_t _width;
	uint8_t _height_bits;
};
} // namespace PVParallelView

#endif
