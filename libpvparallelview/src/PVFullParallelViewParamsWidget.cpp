/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include <pvparallelview/PVFullParallelViewParamsWidget.h>
#include <pvparallelview/PVFullParallelViewSelectionRectangle.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>

#include <QVBoxLayout>
#include <QToolBar>
#include <QCheckBox>
#include <QSignalMapper>
#include <QMenu>
#include <QLineEdit>
#include <QLabel>
#include <QDebug>

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::PVFullParallelViewParamsWidget
 *****************************************************************************/

PVParallelView::PVFullParallelViewParamsWidget::PVFullParallelViewParamsWidget(
    PVFullParallelView* parent)
    : QToolBar(parent)
{
	auto density_action = addAction(QIcon(":/density-axis"), "Density on axes");
	density_action->setCheckable(true);
	density_action->setChecked(false);
	density_action->setShortcut(Qt::Key_D);
	QImage density_legend(60, 1, QImage::Format_ARGB32);
	for (int i = 0; i < density_legend.width(); ++i) {
		density_legend.setPixelColor(
		    i, 0, QColor::fromHsvF((1. - double(i) / density_legend.width()) * 2 / 3., 1., 1.));
	}
	auto density_legend_label = new QLabel();
	density_legend_label->setPixmap(
	    QPixmap::fromImage(density_legend).scaled(density_legend.width(), 16));
	auto dll_action = addWidget(density_legend_label);
	dll_action->setVisible(false);
	connect(density_action, &QAction::toggled, [this, dll_action](bool pushed) {
		auto scene = static_cast<PVParallelView::PVFullParallelScene*>(parent_fpv()->scene());
		scene->get_lines_view().set_axis_width(pushed ? 21 : 3);
		scene->enable_density_on_axes(pushed);
		scene->update_number_of_zones_async();
		dll_action->setVisible(pushed);
		adjustSize();
	});
	setVisible(true);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::update_widgets
 *****************************************************************************/

void PVParallelView::PVFullParallelViewParamsWidget::update_widgets() {}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::set_selection_mode
 *****************************************************************************/

void PVParallelView::PVFullParallelViewParamsWidget::set_selection_mode(int mode)
{
	PVSelectionRectangle::update_selection_mode_selector(_sel_mode_button, mode);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::parent_hcv
 *****************************************************************************/

PVParallelView::PVFullParallelView*
PVParallelView::PVFullParallelViewParamsWidget::parent_fpv() const
{
	assert(qobject_cast<PVFullParallelView*>(parentWidget()));
	return static_cast<PVFullParallelView*>(parentWidget());
}
