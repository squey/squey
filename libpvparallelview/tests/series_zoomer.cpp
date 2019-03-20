/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include <pvkernel/core/inendi_assert.h>

#include <pvparallelview/PVSeriesViewZoomer.h>
#include <pvparallelview/common.h>

#include <QApplication>

#include "common.h"

using namespace PVParallelView;
using Zoom = PVViewZoomer::Zoom;

bool operator==(Zoom const& a, Zoom const& b)
{
	return a.minX == b.minX && a.maxX == b.maxX && a.minY == b.minY && a.maxY == b.maxY;
}

std::ostream& operator<<(std::ostream& out, Zoom const& z)
{
	return out << "minX:" << z.minX << ", maxX:" << z.maxX << ", minY:" << z.minY
	           << ", maxY:" << z.maxY;
}

Zoom mk_zoom(double minX, double maxX, double minY, double maxY)
{
	return {minX, maxX, minY, maxY};
}

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);

	PVViewZoomer vz;
	vz.resize(1000, 1000);
	PV_VALID(vz.current_zoom(), mk_zoom(0., 1., 0., 1.));
	vz.zoom_in(QRect{400, 0, 200, 1000});
	PV_VALID(vz.current_zoom(), mk_zoom(0.4, 0.6, 0., 1.));
}