//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// The heights the crosshair of a series view reads, as percentages of the full
// scale: precise enough to tell two rows of pixels apart, and not more.

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVSeriesViewZoomer.h>

#include <string>

using PVParallelView::PVSeriesViewZoomer;

static std::string label(PVSeriesViewZoomer::zoom_f ratio, PVSeriesViewZoomer::zoom_f row_height)
{
	return PVSeriesViewZoomer::ordinate_label(ratio, row_height).toStdString();
}

int main()
{
	// A row of pixels out of 251 is 0.4 % of the height: one decimal tells two of them apart.
	PV_VALID(label(163.L / 251, 1.L / 251), std::string("64.9%"));
	PV_VALID(label(162.L / 251, 1.L / 251), std::string("64.5%"));

	// Zoomed in a thousand times, it takes four.
	PV_VALID(label(163.L / 251, 1.L / 251 / 1000), std::string("64.9402%"));

	// Rows of 2 % take none.
	PV_VALID(label(1, 1.L / 50), std::string("100%"));

	return 0;
}
