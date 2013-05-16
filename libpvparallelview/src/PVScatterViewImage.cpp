/**
 * \file PVScatterViewImage.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVScatterViewImage.h>

#include <QImage>
#include <QPainter>

#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/picviz_bench.h>

PVParallelView::PVScatterViewImage::PVScatterViewImage()
{
	_hsv_image = new PVCore::PVHSVColor[image_width*image_height];
	_rgb_image = new QImage(image_width, image_height, QImage::Format_RGB32);
}

PVParallelView::PVScatterViewImage::~PVScatterViewImage()
{
	delete [] _hsv_image;
	delete _rgb_image;
}

void PVParallelView::PVScatterViewImage::clear()
{
	memset(_hsv_image, HSV_COLOR_BLACK, image_width*image_height*sizeof(PVCore::PVHSVColor));
}

void PVParallelView::PVScatterViewImage::convert_image_from_hsv_to_rgb()
{
	BENCH_START(image_convertion);
	QRgb* image_rgb = (QRgb*) &_rgb_image->scanLine(0)[0];
#pragma omp parallel for schedule(static, 16)
	for (uint32_t i = 0; i < image_width*image_height; i++) {
		_hsv_image[i].to_rgb((uint8_t*) &image_rgb[i]);
	}
	BENCH_END(image_convertion, "image_convertion", image_width*image_height, sizeof(PVCore::PVHSVColor), image_width*image_height, sizeof(QRgb));
}
