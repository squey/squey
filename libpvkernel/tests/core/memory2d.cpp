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

constexpr int x_offset = -100;
constexpr int y_offset = -150;

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	QImage image(image_width, image_height, QImage::Format_ARGB32);
	PVCore::PVHSVColor* hsv_image = new PVCore::PVHSVColor[image_width*image_height*sizeof(PVCore::PVHSVColor)];
#pragma omp parallel for
	for (PVRow i=0; i<image_width*image_height; i++){
		hsv_image[i].h() = (i/2048)% ((1<<HSV_COLOR_NBITS_ZONE)*6);
	}

	PVCore::memmove2d(hsv_image, image_width, image_height, x_offset, y_offset);

	PVCore::PVHSVColor::to_rgba(hsv_image, image);

	QLabel imageLabel;
	imageLabel.setPixmap(QPixmap::fromImage(image));
	imageLabel.show();

	return app.exec();
}
