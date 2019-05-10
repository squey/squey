/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/widgets/PVHelpWidget.h>

#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelViewParamsWidget.h>

#include <QScrollBar>
#include <QPainter>

PVParallelView::PVZoomedParallelView::PVZoomedParallelView(
    Inendi::PVAxesCombination const& axes_comb, QWidget* parent)
    : PVWidgets::PVGraphicsView(parent)
{
	setMinimumHeight(300);

	install_default_scene_interactor();

	_help_widget = new PVWidgets::PVHelpWidget(this);
	_help_widget->hide();

	_help_widget->initTextFromFile("zoomed parallel view's help", ":help-style");
	_help_widget->addTextFromFile(":help-selection");
	_help_widget->addTextFromFile(":help-layers");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-lines");
	_help_widget->addTextFromFile(":help-application");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-zoomed-parallel-view");
	_help_widget->finalizeText();

	_params_widget = new PVZoomedParallelViewParamsWidget(axes_comb, this);
	_params_widget->adjustSize();
}

PVParallelView::PVZoomedParallelView::~PVZoomedParallelView()
{
	if (get_scene()) {
		delete get_scene();
	}
}

void PVParallelView::PVZoomedParallelView::resizeEvent(QResizeEvent* event)
{
	PVWidgets::PVGraphicsView::resizeEvent(event);

	if (auto zps = static_cast<PVParallelView::PVZoomedParallelScene*>(get_scene())) {
		_params_widget->move({0, 0});
		_params_widget->adjustSize();
		_params_widget->raise();

		bool need_recomputation = event->oldSize().height() != event->size().height();
		zps->resize_display(need_recomputation);
	}
}
