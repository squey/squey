/**
 * \file PVScatterViewImage.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVScatterViewImage.h>

#include <QImage>
#include <QPainter>

#include <pvkernel/core/PVHSVColor.h>

PVParallelView::PVScatterViewImage::PVScatterViewImage()
{
	_hsv_image = new PVCore::PVHSVColor[image_width*image_height];
	_rgb_image = new QImage(image_width, image_height, QImage::Format_ARGB32);
}

PVParallelView::PVScatterViewImage::~PVScatterViewImage()
{
	delete [] _hsv_image;
	delete _rgb_image;
}

void PVParallelView::PVScatterViewImage::clear()
{
	memset(_hsv_image, HSV_COLOR_TRANSPARENT, image_width*image_height*sizeof(PVCore::PVHSVColor));
}

void PVParallelView::PVScatterViewImage::convert_image_from_hsv_to_rgb()
{
	PVCore::PVHSVColor::to_rgba(_hsv_image, *_rgb_image);
}
