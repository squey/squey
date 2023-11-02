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

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>

#include <pvparallelview/PVDisplayViewTimeseries.h>
#include <pvparallelview/PVSeriesViewWidget.h>

#include <pvkernel/widgets/PVModdedIcon.h>

PVDisplays::PVDisplayViewTimeseries::PVDisplayViewTimeseries()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu | PVDisplayIf::HasHelpPage,
                      "Series view",
                      PVModdedIcon("series"),
                      "New series view")
{
}

QWidget* PVDisplays::PVDisplayViewTimeseries::create_widget(Squey::PVView* view,
                                                            QWidget* parent,
                                                            Params const& params) const
{
	return new PVParallelView::PVSeriesViewWidget(view, col_param(view, params, 0), parent);
}

bool PVDisplays::PVDisplayViewTimeseries::abscissa_filter(Squey::PVView* view, PVCol axis) const
{
	return view->get_axes_combination().get_axis(axis).get_type().left(4) == "time" or
	       view->get_axes_combination().get_axis(axis).get_type().left(8) == "duration" or
	       view->get_axes_combination().get_axis(axis).get_type().left(7) == "number_";
}

void PVDisplays::PVDisplayViewTimeseries::add_to_axis_menu(
    QMenu& menu,
    PVCol axis,
    PVCombCol axis_comb,
    Squey::PVView* view,
    PVDisplays::PVDisplaysContainer* container)
{
	if (abscissa_filter(view, axis)) {
		PVDisplayViewIf::add_to_axis_menu(menu, axis, axis_comb, view, container);
	}
}
