//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
