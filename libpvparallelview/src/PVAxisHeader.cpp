/**
 * \file PVAxisHeader.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVAxisHeader.h>
#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVAxisLabel.h>

#include <pvdisplays/PVDisplaysImpl.h>
#include <pvdisplays/PVDisplaysContainer.h>

#include <picviz/PVView.h>

#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/core/PVAlgorithms.h>

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

PVParallelView::PVAxisHeader::PVAxisHeader(
	const Picviz::PVView& view,
	PVSlidersGroup* sg,
	PVAxisGraphicsItem* parent) :
	QGraphicsRectItem(parent),
	_view(view),
	_sliders_group(sg),
	_axis_selected_animation(new __impl::PVAxisSelectedAnimation(this))
{
	setAcceptHoverEvents(true); // This is needed to enable hover events
	setCursor(Qt::ArrowCursor);
	setPen(QPen(Qt::NoPen));
	setBrush(QBrush(Qt::NoBrush));
}

void PVParallelView::PVAxisHeader::paint(
	QPainter* painter,
	const QStyleOptionGraphicsItem* option,
	QWidget* widget
)
{
	QGraphicsRectItem::paint(painter, option, widget);
}

void PVParallelView::PVAxisHeader::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
	// Get parent PVDisplaysContainer if available
	QList<QGraphicsView*> parent_views = scene()->views();
	assert(parent_views.size() == 1);

	QWidget* parent_view = parent_views.at(0);
	PVDisplays::PVDisplaysContainer* container = PVCore::get_qobject_parent_of_type<PVDisplays::PVDisplaysContainer*>(parent_view);

	QMenu menu;

	if (container) {
		PVDisplays::get().add_displays_view_axis_menu(menu, container, SLOT(create_view_axis_widget()), (Picviz::PVView*) &_view, get_axis_index());
		if (!is_last_axis()) {
			PVDisplays::get().add_displays_view_zone_menu(menu, container, SLOT(create_view_zone_widget()), (Picviz::PVView*) &_view, get_axis_index());
		}
		menu.addSeparator();
	}
	QAction *ars = menu.addAction("New selection cursors");
	connect(ars, SIGNAL(triggered()), this, SLOT(new_selection_sliders()));

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
	_width = width;
	setRect(0, -108, width, 100);
}

void PVParallelView::PVAxisHeader::hoverEnterEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	emit mouse_hover_entered(get_axis_index(), true);
}

void PVParallelView::PVAxisHeader::hoverLeaveEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	emit mouse_hover_entered(get_axis_index(), false);
}

void PVParallelView::PVAxisHeader::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	// store mouse coordinates
	_clicking_pos = event->pos();
	event->accept();
}

void PVParallelView::PVAxisHeader::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->pos() == _clicking_pos) { // Should use manhattan distance...
		event->accept(); // Prevent the scene from handling this event
		emit mouse_clicked(get_axis_index());
	}
}

PVCol PVParallelView::PVAxisHeader::get_axis_index() const
{
	return _view.get_axes_combination().get_index_by_id(_sliders_group->get_axis_id());
}

PVParallelView::PVAxisGraphicsItem* PVParallelView::PVAxisHeader::axis()
{
	return dynamic_cast<PVAxisGraphicsItem*>(parentItem());
}

PVParallelView::PVAxisGraphicsItem const* PVParallelView::PVAxisHeader::axis()  const
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
	/*QDialog *view_dlg = new QDialog();

	view_dlg->setMaximumWidth(1024);
	view_dlg->setMaximumHeight(1024);
	view_dlg->setAttribute(Qt::WA_DeleteOnClose, true);

	QLayout *view_layout = new QVBoxLayout(view_dlg);
	view_layout->setContentsMargins(0, 0, 0, 0);
	view_dlg->setLayout(view_layout);

	QWidget *view = common::get_lib_view(const_cast<Picviz::PVView&>(_lib_view))->create_zoomed_view(_axis_index);

	view_layout->addWidget(view);
	view_dlg->show();*/

	emit new_zoomed_parallel_view(get_axis_index());
}


void PVParallelView::PVAxisHeader::new_selection_sliders()
{
	_sliders_group->add_selection_sliders(0, 1024);
}

/******************************************************************************
 *
 * PVParallelView::__impl::PVAxisSelectedAnimation
 *
 *****************************************************************************/

PVParallelView::__impl::PVAxisSelectedAnimation::PVAxisSelectedAnimation(PVAxisHeader* parent) :
	QObject(parent)
{
	// Setup opacity animation
	_opacity_animation = new QPropertyAnimation(this, "opacity");
	_opacity_animation->setStartValue(opacity_animation_min_amount);
	_opacity_animation->setEndValue(opacity_animation_max_amount);
	_opacity_animation->setDuration(opacity_animation_duration_ms);
	_opacity_animation->setEasingCurve(QEasingCurve::InOutQuad);
	QGraphicsOpacityEffect* opacity_effect = new QGraphicsOpacityEffect();
	//_selected_axis_hole->setGraphicsEffect(opacity_effect);
	header()->setGraphicsEffect(opacity_effect);

	_title_highlight = new PVGraphicsPolygonItem();

	QRectF local_bounding_rect = header()->axis()->_label->boundingRect();
	qreal min_width = 5;
	qreal min_height = 15;
	local_bounding_rect.setWidth(std::max(local_bounding_rect.width(), min_width));
	local_bounding_rect.setHeight(std::max(local_bounding_rect.height(), min_height));
	QRectF transformed_bounding_rect = header()->axis()->_label->mapToParent(local_bounding_rect).boundingRect();

	qreal a = 8;   // margin under and over text
	qreal b = a;   // thickness
	qreal c = 5;   // margin after text
	qreal e = 5;   // bevel
	qreal d = c;   // width of the shape after the end of the text
	qreal y_trans = 12;

	qreal label_width = transformed_bounding_rect.width();
	qreal label_height = transformed_bounding_rect.height();
	int label_length = cos(PVAxisGraphicsItem::label_rotation)*(local_bounding_rect.height()*3);

	qreal x0 = 0;
	qreal y0 = -a;

	qreal x1 = x0;
	qreal y1 = y0 -b;

	qreal x2 = label_width -(a/2) +c -b -e +d;
	qreal y2 = -label_height + (a/2) -c -a -e -d;

	qreal x3 = x2 +b +label_length +a -2*b +2*e;
	qreal y3 = y2 +a +label_length -b +2*e;

	qreal x4 = a+c+e+b/2;
	qreal y4 = +c+e-b/2;

	qreal x5 = a+c+e+b/2-b;
	qreal y5 = +c+e-b/2;

	qreal x6 = x3 -b -e -d;
	qreal y6 = y3 -e +d;

	qreal x7 = x2 +e -d;
	qreal y7 = y2 +b +e +d;

	QPolygonF polygon;
	polygon << QPointF(x0, y0);
	polygon << QPointF(x1, y1);
	polygon << QPointF(x2, y2);
	polygon << QPointF(x3, y3);
	polygon << QPointF(x4, y4);
	polygon << QPointF(x5, y5);
	polygon << QPointF(x6, y6);
	polygon << QPointF(x7, y7);
	polygon.translate(0, -y_trans);

	_title_highlight->setPolygon(polygon);
	_title_highlight->setBrush(header()->axis()->lib_axis()->get_titlecolor().toQColor());
	_title_highlight->setPen(Qt::NoPen);
	_title_highlight->setVisible(false);
	QGraphicsOpacityEffect* opacity_effect2 = new QGraphicsOpacityEffect();
	opacity_effect2->setOpacity(0.5);
	_title_highlight->setGraphicsEffect(opacity_effect2);
	header()->axis()->addToGroup(_title_highlight);
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
	}
	else {
		_title_highlight->setVisible(false);
		_opacity_animation->setDirection(QAbstractAnimation::Backward);
		_opacity_animation->start();
	}
}

void PVParallelView::__impl::PVAxisSelectedAnimation::set_opacity(qreal opacity)
{
	QGraphicsOpacityEffect* opacity_effect1 = (QGraphicsOpacityEffect*) _title_highlight->graphicsEffect();
	opacity_effect1->setOpacity(opacity);
	QGraphicsOpacityEffect* opacity_effect2 = (QGraphicsOpacityEffect*) header()->graphicsEffect();
	opacity_effect2->setOpacity(opacity);
}


void PVParallelView::__impl::PVGraphicsPolygonItem::paint(
	QPainter* painter,
	const QStyleOptionGraphicsItem* option,
	QWidget* widget)
{
	painter->setRenderHint(QPainter::Antialiasing, true);
	painter->setRenderHint(QPainter::HighQualityAntialiasing, true);
	QGraphicsPolygonItem::paint(painter, option, widget);
}
