/**
 * \file PVScatterViewImage.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVSCATTERVIEWIMAGE_H__
#define __PVSCATTERVIEWIMAGE_H__

#include <cstddef>
#include <cstdint>

#include <boost/utility.hpp>

#include <QRect>
#include <QImage>

class QPainter;

namespace PVCore {
class PVHSVColor;
}

namespace PVParallelView {

class PVZoomedZoneTree;

class PVScatterViewImage: boost::noncopyable
{
public:
	constexpr static uint32_t image_width = 2048;
	constexpr static uint32_t image_height = image_width;

public:
	PVScatterViewImage();
	PVScatterViewImage(PVScatterViewImage&& o)
	{
		move(o);
	}

	~PVScatterViewImage();

public:
	void clear(const QRect& rect = QRect());

	void convert_image_from_hsv_to_rgb();

	PVCore::PVHSVColor* get_hsv_image() { return _hsv_image; }
	QImage& get_rgb_image() { return _rgb_image; };

	const PVCore::PVHSVColor* get_hsv_image() const { return _hsv_image; }
	const QImage& get_rgb_image()  const { return _rgb_image; };

public:
	PVScatterViewImage& operator=(PVScatterViewImage&& o)
	{
		if (&o != this) {
			move(o);
		}
		return *this;
	}

private:
	inline void move(PVScatterViewImage& o)
	{
		_hsv_image = o._hsv_image;
		_rgb_image = o._rgb_image;
		o._hsv_image = nullptr;
	}

private:
	PVCore::PVHSVColor* _hsv_image;
	QImage _rgb_image;
};

}

#endif // __PVSCATTERVIEWIMAGE_H__
