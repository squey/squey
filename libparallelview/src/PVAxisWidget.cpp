
#include <iostream>

// pour PVParallelView::AxisWidth
#include <pvparallelview/common.h>

#include <pvparallelview/PVAxisWidget.h>

#include <QPainter>

PVParallelView::PVAxisWidget::PVAxisWidget(const Picviz::PVAxis &axis)
{
	_text = axis.get_name();

	update_bbox();
}

PVParallelView::PVAxisWidget::PVAxisWidget(const QString &text) :
	_text(text)
{
	std::cout << "adding " << qPrintable(text) << std::endl;
	update_bbox();
}

QRectF PVParallelView::PVAxisWidget::boundingRect () const
{
	return _bbox;
}

void PVParallelView::PVAxisWidget::paint(QPainter *painter,
                                         const QStyleOptionGraphicsItem */*option*/,
                                         QWidget */*widget*/)
{
	painter->fillRect(x() - PVParallelView::AxisWidth, y(),
	                  PVParallelView::AxisWidth, IMAGE_HEIGHT,
	                  Qt::SolidPattern);
	painter->save();
	painter->translate(x() - PVParallelView::AxisWidth, y());
	painter->rotate(-45.);
	painter->drawText(10, 0, _text);
	painter->restore();
}

void PVParallelView::PVAxisWidget::update_bbox()
{
	QRectF bbox = QRectF(x() - PVParallelView::AxisWidth, y(),
	                     PVParallelView::AxisWidth, IMAGE_HEIGHT);

	// des valeurs au pif (mais pas au rouge)
	_bbox = bbox.united(QRectF(x() - PVParallelView::AxisWidth, y(), 200, 100));
}
