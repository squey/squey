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

#include <QScrollBar64>
#include <QPainter>

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::PVZoomedParallelView
 *****************************************************************************/

PVParallelView::PVZoomedParallelView::PVZoomedParallelView(QWidget* parent)
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
	_help_widget->addTextFromFile(":help-mouse-zoomed-paralllel-view");
	_help_widget->finalizeText();

	_params_widget = new PVZoomedParallelViewParamsWidget(this);
	_params_widget->setStyleSheet("QToolBar {" + frame_qss_bg_color + "}");
	_params_widget->setAutoFillBackground(true);
	_params_widget->adjustSize();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::~PVZoomedParallelView
 *****************************************************************************/

PVParallelView::PVZoomedParallelView::~PVZoomedParallelView()
{
	if (get_scene()) {
		delete get_scene();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::resizeEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelView::resizeEvent(QResizeEvent* event)
{
	PVWidgets::PVGraphicsView::resizeEvent(event);

	PVParallelView::PVZoomedParallelScene* zps =
	    (PVParallelView::PVZoomedParallelScene*)get_scene();
	if (zps == nullptr) {
		return;
	}

	bool need_recomputation = event->oldSize().height() != event->size().height();

	QPoint pos(get_viewport()->width() - frame_offsets.right(), frame_offsets.top());

	pos -= QPoint(_params_widget->width(), 0);
	_params_widget->move(pos);
	_params_widget->raise();

	zps->resize_display(need_recomputation);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::drawForeground
 *****************************************************************************/

void PVParallelView::PVZoomedParallelView::drawForeground(QPainter* painter, const QRectF& rect)
{
	PVGraphicsView::drawForeground(painter, rect);

	painter->save();

	QFont f(painter->font());
	f.setWeight(QFont::Bold);
	painter->setFont(f);

	QFontMetrics fm = painter->fontMetrics();
	const QSize text_size = fm.size(Qt::TextSingleLine, _display_axis_name);
	const QRect frame(frame_offsets.left(), frame_offsets.top(),
	                  text_size.width() + frame_margins.left() + frame_margins.right(),
	                  text_size.height() + frame_margins.top() + frame_margins.bottom());

	painter->setPen(Qt::NoPen);
	painter->setBrush(frame_bg_color);
	painter->drawRect(frame);

	painter->setPen(QPen(frame_text_color, 0));
	painter->setBrush(Qt::NoBrush);

	const QPoint text_pos(frame.left() + frame_margins.left(),
	                      frame.top() + frame_margins.top() + fm.ascent());

	painter->drawText(text_pos, _display_axis_name);

	painter->restore();
}
