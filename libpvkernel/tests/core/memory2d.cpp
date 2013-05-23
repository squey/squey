/**
 * \file memory2d.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <QApplication>
#include <QImage>
#include <QRgb>
#include <QLabel>

#include <pvkernel/core/PVMemory2D.h>
#include <pvkernel/core/PVHSVColor.h>

constexpr uint32_t image_width = 500;
constexpr uint32_t image_height = 600;

constexpr int x_offset = -150;
constexpr int y_offset = 177;

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	QImage image(image_width, image_height, QImage::Format_ARGB32);
	PVCore::PVHSVColor* hsv_image = new PVCore::PVHSVColor[image_width*image_height*sizeof(PVCore::PVHSVColor)];

#pragma omp parallel for
	for (PVRow j=0; j<image_height; j++) {
		PVCore::PVHSVColor color = HSV_COLOR_BLACK;
		if ((j % 2) == 0) {
			color = ((j*image_width)/2048) % ((1<<HSV_COLOR_NBITS_ZONE)*6);
		}
		memset(&hsv_image[j*image_width], color.h(), image_width);
	}

	PVCore::memmove2d(hsv_image, image_width, image_height, x_offset, y_offset);

	PVCore::PVHSVColor::to_rgba(hsv_image, image);

	QLabel imageLabel;
	imageLabel.setPixmap(QPixmap::fromImage(image));
	imageLabel.show();

	return app.exec();
}
