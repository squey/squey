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

#include <pvparallelview/PVZoomedParallelViewParamsWidget.h>
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/common.h>

#include <QAction>
#include <QMenu>

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewParamsWidget::PVZoomedParallelViewParamsWidget
 *****************************************************************************/

PVParallelView::PVZoomedParallelViewParamsWidget::PVZoomedParallelViewParamsWidget(
    Inendi::PVAxesCombination const& axes_comb, QWidget* parent)
    : QToolBar(parent)
{
	setIconSize(QSize(17, 17));
	setStyleSheet("QToolBar {" + frame_qss_bg_color + "} QComboBox { font-weight: bold; color: " +
	              frame_text_color.name(QColor::HexArgb) + "; }");
	setAutoFillBackground(true);

	_menu = new PVWidgets::PVAxisComboBox(axes_comb,
	                                      PVWidgets::PVAxisComboBox::AxesShown::CombinationAxes);
	addWidget(_menu);

	connect(_menu, &PVWidgets::PVAxisComboBox::current_axis_changed, this,
	        [this](PVCol, PVCombCol comb_col) { change_to_col(comb_col); });
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewParamsWidget::build_axis_menu
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewParamsWidget::build_axis_menu(PVCombCol active_axis)
{
	blockSignals(true);
	_menu->refresh_axes();
	_menu->set_current_axis(active_axis);
	blockSignals(false);
}
