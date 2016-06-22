/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVAbstractAxisSliders.h>
#include <pvparallelview/PVParallelView.h>

#include <QGraphicsSimpleTextItem>
#include <QPainter>

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSliders::PVAbstractAxisSliders
 *****************************************************************************/

PVParallelView::PVAbstractAxisSliders::PVAbstractAxisSliders(QGraphicsItem* parent,
                                                             PVSlidersManager* sm_p,
                                                             PVSlidersGroup* group,
                                                             const char* text)
    : QObject(nullptr), QGraphicsItemGroup(parent), _sliders_manager_p(sm_p), _group(group)
{
	setHandlesChildEvents(false);

	/* RH: the 4 ' ' are a hacky visual offset to display the text at the
	 * left of the axis without overlapping the sliders
	 */
	_text = new QGraphicsSimpleTextItem(QString("    ") + text);
	addToGroup(_text);

	_text->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
	_text->setBrush(Qt::white);
	_text->hide();
}

void PVParallelView::PVAbstractAxisSliders::paint(QPainter* painter,
                                                  const QStyleOptionGraphicsItem* option,
                                                  QWidget* widget)
{
	QGraphicsItemGroup::paint(painter, option, widget);

#ifdef INENDI_DEVELOPER_MODE
	if (common::show_bboxes()) {
		painter->save();
		painter->setPen(QPen(QColor(0xFF, 0xFF, 0), 0));
		painter->setBrush(QColor(0xFF, 0xFF, 0, 128));
		const QRectF br = boundingRect();
		painter->drawRect(br);
		painter->restore();
	}
#endif
}

QRectF PVParallelView::PVAbstractAxisSliders::boundingRect() const
{
	return childrenBoundingRect();
}
