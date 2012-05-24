
#include <iostream>

#include <picviz/PVAxis.h>
#include <pvparallelview/PVAxisWidget.h>

// pour PVParallelView::AxisWidth
#include <pvparallelview/common.h>

#include <QPainter>
#include <QGraphicsScene>

// Used to draw the axis out of the image zone
#define PVAW_CST 8

/*****************************************************************************
 * PVParallelView::PVAxisWidget::PVAxisWidget
 *****************************************************************************/

PVParallelView::PVAxisWidget::PVAxisWidget(Picviz::PVAxis *axis) :
	_axis(axis)
{
}

/*****************************************************************************
 * PVParallelView::PVAxisWidget::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVAxisWidget::boundingRect () const
{
	QRectF bbox = QRectF(- PVParallelView::AxisWidth, - PVAW_CST,
	                     PVParallelView::AxisWidth, IMAGE_HEIGHT + (2 * PVAW_CST));

	return bbox.united(QRectF(- PVParallelView::AxisWidth, 0, 50, -50));
}

/*****************************************************************************
 * PVParallelView::PVAxisWidget::paint
 *****************************************************************************/

void PVParallelView::PVAxisWidget::paint(QPainter *painter,
                                         const QStyleOptionGraphicsItem */*option*/,
                                         QWidget */*widget*/)
{
	QPen pen = painter->pen();

	painter->fillRect(0, - PVAW_CST,
	                  PVParallelView::AxisWidth, IMAGE_HEIGHT + (2 * PVAW_CST),
	                  _axis->get_color().toQColor());
	painter->save();
	painter->translate(- PVParallelView::AxisWidth, - PVAW_CST);
	painter->rotate(-45.);
	painter->setPen(_axis->get_titlecolor().toQColor());
	painter->drawText(10, 0, _axis->get_name());
	painter->setPen(pen);
	painter->restore();
}

/*****************************************************************************
 * PVParallelView::PVAxisWidget::add_range_sliders
 *****************************************************************************/

void PVParallelView::PVAxisWidget::add_range_sliders(uint32_t p1, uint32_t p2)
{
	PVParallelView::PVAxisRangeSliders sliders;

	sliders.first = new PVParallelView::PVAxisSlider(0, 1023, p1);
	sliders.second = new PVParallelView::PVAxisSlider(0, 1023, p2);

	sliders.first->setPos(pos());
	sliders.second->setPos(pos());

	addToGroup(sliders.first);
	addToGroup(sliders.second);

	_sliders.push_back(sliders);
}
