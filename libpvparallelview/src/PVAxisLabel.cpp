/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVView.h>

#include <pvparallelview/PVAxisLabel.h>
#include <pvparallelview/PVAxisGraphicsItem.h>

#include <QDialog>
#include <QLayout>
#include <QMenu>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QFontMetrics>

#include <iostream>

/*****************************************************************************
 * PVParallelView::PVAxisLabel::PVAxisLabel
 *****************************************************************************/

PVParallelView::PVAxisLabel::PVAxisLabel(const Inendi::PVView& view, QGraphicsItem* parent)
    : QGraphicsSimpleTextItem(parent), _lib_view(view)
{
	setFlag(QGraphicsItem::ItemClipsToShape, true);
}

/*****************************************************************************
 * PVParallelView::PVAxisLabel::~PVAxisLabel
 *****************************************************************************/

PVParallelView::PVAxisLabel::~PVAxisLabel()
{
	if (scene()) {
		scene()->removeItem(this);
	}
	if (group()) {
		group()->removeFromGroup(this);
	}
}

void PVParallelView::PVAxisLabel::set_bounding_box_width(int width)
{
	if (width) {
		_bounding_box_width = width;
	}
}

bool PVParallelView::PVAxisLabel::contains(const QPointF& point) const
{
	QRectF rect = QGraphicsSimpleTextItem::boundingRect();
	return rect.contains(point);
}

QRectF PVParallelView::PVAxisLabel::boundingRect() const
{
	return QGraphicsSimpleTextItem::boundingRect();
}

QPainterPath PVParallelView::PVAxisLabel::shape() const
{
	QPainterPath path;
	QRectF rect = QGraphicsSimpleTextItem::boundingRect();
	if (_bounding_box_width) {
		rect.setWidth(_bounding_box_width);
	}
	path.addRect(rect);

	return path;
}

PVParallelView::PVAxisGraphicsItem const* PVParallelView::PVAxisLabel::get_parent_axis() const
{
	return dynamic_cast<PVAxisGraphicsItem const*>(parentItem());
}

void PVParallelView::PVAxisLabel::set_text(const QString& text)
{
	QFontMetrics metrics = QFontMetrics(font());

	if (metrics.horizontalAdvance(text) > MAX_WIDTH) {

		setText(metrics.elidedText(text, Qt::ElideMiddle, MAX_WIDTH));

		setToolTip(text);
	} else {
		setText(text);
	}
}
