/**
 * \file PVZoomedParallelView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/widgets/PVHelpWidget.h>

#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>

#include <QScrollBar64>

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::PVZoomedParallelView
 *****************************************************************************/

PVParallelView::PVZoomedParallelView::PVZoomedParallelView(QWidget *parent) :
	PVWidgets::PVGraphicsView(parent)
{
	setMinimumHeight(300);

	install_default_scene_interactor();

	_help_widget = new PVWidgets::PVHelpWidget(this);
	_help_widget->hide();

	_help_widget->initTextFromFile("zoomed parallel view's help",
	                               ":help-style");
	_help_widget->addTextFromFile(":help-selection");
	_help_widget->addTextFromFile(":help-layers");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-lines");
	_help_widget->addTextFromFile(":help-application");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-zoomed-paralllel-view");
	_help_widget->finalizeText();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::~PVZoomedParallelView
 *****************************************************************************/

PVParallelView::PVZoomedParallelView::~PVZoomedParallelView()
{
	if (get_scene()) {
		get_scene()->deleteLater();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::resizeEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelView::resizeEvent(QResizeEvent *event)
{
	PVWidgets::PVGraphicsView::resizeEvent(event);

	PVParallelView::PVZoomedParallelScene *zps = (PVParallelView::PVZoomedParallelScene*)get_scene();
	if(zps == nullptr) {
		return;
	}

	bool need_recomputation = event->oldSize().height() != event->size().height();

	zps->resize_display(need_recomputation);
}
