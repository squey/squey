/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
