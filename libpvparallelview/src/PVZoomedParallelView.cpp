/**
 * \file PVZoomedParallelView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/widgets/PVHelpWidget.h>

#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelViewParamsWidget.h>

#include <QScrollBar64>
#include <QPainter>

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

	_params_widget = new PVZoomedParallelViewParamsWidget(this);
	_params_widget->setAutoFillBackground(true);
	_params_widget->adjustSize();
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

	QPoint pos = QPoint(get_viewport()->size().width() - 4, 4);
	pos -= QPoint(_params_widget->width(), 0);
	_params_widget->move(pos);
	_params_widget->raise();

	zps->resize_display(need_recomputation);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::drawForeground
 *****************************************************************************/

void PVParallelView::PVZoomedParallelView::drawForeground(QPainter* painter,
                                                          const QRectF& rect)
{
	PVGraphicsView::drawForeground(painter, rect);

	painter->setPen(QPen(QColor(0x16, 0xe8, 0x2a), 0));

	painter->drawText(8, 16, _display_axis_name);
}
