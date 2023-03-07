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

#include <pvparallelview/PVAxisHeader.h>
#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVAxisLabel.h>

#include <pvdisplays/PVDisplayIf.h>
#include <pvdisplays/PVDisplaysContainer.h>

#include <squey/PVView.h>
#include <squey/PVPlotted.h>
#include <squey/PVMapped.h>

#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <QActionGroup>
#include <QApplication>
#include <QClipboard>
#include <QMenu>
#include <QGraphicsView>
#include <QPainter>
#include <QLinearGradient>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsBlurEffect>
#include <QPropertyAnimation>

#include <iostream>

/******************************************************************************
 *
 * PVParallelView::PVAxisHeader::PVAxisHeader
 *
 *****************************************************************************/

PVParallelView::PVAxisHeader::PVAxisHeader(const Squey::PVView& view,
                                           PVCombCol comb_col,
                                           PVAxisGraphicsItem* parent)
    : QGraphicsRectItem(parent)
    , _view(view)
    , _comb_col(comb_col)
    , _axis_selected_animation(new __impl::PVAxisSelectedAnimation(this))
    , _clicked(false)
    , _click_event(QEvent::GraphicsSceneMousePress)
{
	setAcceptHoverEvents(true); // This is needed to enable hover events
	setCursor(Qt::ArrowCursor);
	setPen(QPen(Qt::NoPen));
	setBrush(QBrush(Qt::NoBrush));
}

void PVParallelView::PVAxisHeader::contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{
	QMenu menu;

	if (auto* container =
	        PVCore::get_qobject_parent_of_type<PVDisplays::PVDisplaysContainer*>(event->widget())) {
		PVDisplays::add_displays_view_axis_menu(menu, container, (Squey::PVView*)&_view,
		                                        _comb_col);
		menu.addSeparator();
	}
	QAction* ars = menu.addAction("New selection cursors");
	connect(ars, &QAction::triggered, this, &PVAxisHeader::new_selection_slider);

	if (menu.exec(event->screenPos()) != nullptr) {
		event->accept();
	}
}

void PVParallelView::PVAxisHeader::start(bool start)
{
	_started = start;
	_axis_selected_animation->start(start);
}

void PVParallelView::PVAxisHeader::set_width(int width)
{
	setRect(0, -108, width, 100);
}

void PVParallelView::PVAxisHeader::hoverEnterEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	Q_EMIT mouse_hover_entered(_comb_col, true);
}

void PVParallelView::PVAxisHeader::hoverLeaveEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	Q_EMIT mouse_hover_entered(_comb_col, false);
}

void PVParallelView::PVAxisHeader::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		// mark the beginning of the possible click
		_clicked = true;
		// and save the event to send it later
		_click_event.setWidget(event->widget());
		_click_event.setButtonDownScenePos(event->button(),
		                                   event->buttonDownScenePos(event->button()));
		_click_event.setButtonDownScreenPos(event->button(),
		                                    event->buttonDownScreenPos(event->button()));

		_click_event.setPos(event->pos());
		_click_event.setScenePos(event->scenePos());
		_click_event.setScreenPos(event->screenPos());
		_click_event.setLastScenePos(event->lastScenePos());
		_click_event.setLastScreenPos(event->lastScreenPos());

		_click_event.setButtons(event->buttons());
		_click_event.setButton(event->button());
		_click_event.setModifiers(event->modifiers());
		_click_event.setAccepted(false);
	}
}

void PVParallelView::PVAxisHeader::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	if ((event->button() == Qt::LeftButton) && _clicked) {
		event->accept(); // Prevent the scene from handling this event
		_view.set_axis_clicked(_comb_col);
		_clicked = false;
	} else {
		event->ignore();
	}
}

void PVParallelView::PVAxisHeader::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	if (_clicked) {
		_clicked = false;
		Qt::MouseButtons b = acceptedMouseButtons();
		setAcceptedMouseButtons(Qt::NoButton);
		QApplication::sendEvent(scene(), &_click_event);
		setAcceptedMouseButtons(b);
	}
	QGraphicsRectItem::mouseMoveEvent(event);
}

PVParallelView::PVAxisGraphicsItem* PVParallelView::PVAxisHeader::axis()
{
	return dynamic_cast<PVAxisGraphicsItem*>(parentItem());
}

PVParallelView::PVAxisGraphicsItem const* PVParallelView::PVAxisHeader::axis() const
{
	return dynamic_cast<PVAxisGraphicsItem const*>(parentItem());
}

bool PVParallelView::PVAxisHeader::is_last_axis() const
{
	PVAxisGraphicsItem const* parent_axis = axis();
	if (parent_axis) {
		return parent_axis->is_last_axis();
	}
	return false;
}

void PVParallelView::PVAxisHeader::new_zoomed_parallel_view()
{
	Q_EMIT new_zoomed_parallel_view(_comb_col);
}

/******************************************************************************
 *
 * PVParallelView::__impl::PVAxisSelectedAnimation
 *
 *****************************************************************************/

PVParallelView::__impl::PVAxisSelectedAnimation::PVAxisSelectedAnimation(PVAxisHeader* parent)
    : QObject(parent)
{
	// Setup opacity animation
	_opacity_animation = new QPropertyAnimation(this, "opacity");
	_opacity_animation->setStartValue(opacity_animation_min_amount);
	_opacity_animation->setEndValue(opacity_animation_max_amount);
	_opacity_animation->setDuration(opacity_animation_duration_ms);
	_opacity_animation->setEasingCurve(QEasingCurve::InOutQuad);
	auto* opacity_effect = new QGraphicsOpacityEffect();
	//_selected_axis_hole->setGraphicsEffect(opacity_effect);
	header()->setGraphicsEffect(opacity_effect);

	_title_highlight = new PVGraphicsPolygonItem();
	_title_highlight->setFlags(QGraphicsItem::ItemIgnoresTransformations);

	QRectF local_bounding_rect = header()->axis()->label()->boundingRect();
	qreal min_width = 5;
	qreal min_height = 15;
	local_bounding_rect.setWidth(std::max(local_bounding_rect.width(), min_width));
	local_bounding_rect.setHeight(std::max(local_bounding_rect.height(), min_height));
	QRectF transformed_bounding_rect =
	    header()->axis()->label()->mapToParent(local_bounding_rect).boundingRect();

	qreal a = 8; // margin under and over text
	qreal b = a; // thickness
	qreal c = 5; // margin after text
	qreal e = 5; // bevel
	qreal d = c; // width of the shape after the end of the text
	qreal y_trans = PVAxisGraphicsItem::axis_extend - 4;

	qreal label_width = transformed_bounding_rect.width();
	qreal label_height = transformed_bounding_rect.height();
	int label_length = cos(PVAxisGraphicsItem::label_rotation) * (local_bounding_rect.height() * 3);

	qreal x0 = 0;
	qreal y0 = -a;

	qreal x1 = x0;
	qreal y1 = y0 - b;

	qreal x2 = label_width - (a / 2) + c - b - e + d;
	qreal y2 = -label_height + (a / 2) - c - a - e - d;

	qreal x3 = x2 + b + label_length + a - 2 * b + 2 * e;
	qreal y3 = y2 + a + label_length - b + 2 * e;

	qreal x4 = a + c + e + b / 2;
	qreal y4 = +c + e - b / 2;

	qreal x5 = a + c + e + b / 2 - b;
	qreal y5 = +c + e - b / 2;

	qreal x6 = x3 - b - e - d;
	qreal y6 = y3 - e + d;

	qreal x7 = x2 + e - d;
	qreal y7 = y2 + b + e + d;

	QPolygonF polygon;
	polygon << QPointF(x0, y0);
	polygon << QPointF(x1, y1);
	polygon << QPointF(x2, y2);
	polygon << QPointF(x3, y3);
	polygon << QPointF(x4, y4);
	polygon << QPointF(x5, y5);
	polygon << QPointF(x6, y6);
	polygon << QPointF(x7, y7);
	polygon.translate(0, y_trans);

	_title_highlight->setPolygon(polygon);
	_title_highlight->setBrush(header()->axis()->get_title_color());
	_title_highlight->setPen(Qt::NoPen);
	_title_highlight->setVisible(false);
	auto* opacity_effect2 = new QGraphicsOpacityEffect();
	opacity_effect2->setOpacity(0.5);
	_title_highlight->setGraphicsEffect(opacity_effect2);
	_title_highlight->setParentItem(header()->axis()->label());
}

PVParallelView::__impl::PVAxisSelectedAnimation::~PVAxisSelectedAnimation()
{
	delete _opacity_animation;
}

void PVParallelView::__impl::PVAxisSelectedAnimation::start(bool start)
{
	if (start) {
		_title_highlight->setVisible(true);
		_opacity_animation->setDirection(QAbstractAnimation::Forward);
		_opacity_animation->start();
	} else {
		_title_highlight->setVisible(false);
		_opacity_animation->setDirection(QAbstractAnimation::Backward);
		_opacity_animation->start();
	}
}

void PVParallelView::__impl::PVAxisSelectedAnimation::set_opacity(qreal opacity)
{
	auto* opacity_effect1 =
	    (QGraphicsOpacityEffect*)_title_highlight->graphicsEffect();
	opacity_effect1->setOpacity(opacity);
	auto* opacity_effect2 = (QGraphicsOpacityEffect*)header()->graphicsEffect();
	opacity_effect2->setOpacity(opacity);
}

void PVParallelView::__impl::PVGraphicsPolygonItem::paint(QPainter* painter,
                                                          const QStyleOptionGraphicsItem* option,
                                                          QWidget* widget)
{
	painter->setRenderHint(QPainter::Antialiasing, true);
	QGraphicsPolygonItem::paint(painter, option, widget);
}
