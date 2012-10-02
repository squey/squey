
#include <pvparallelview/PVAbstractRangeAxisSliders.h>
#include <pvparallelview/PVSlidersGroup.h>

#include <QPainter>

/*****************************************************************************
 * PVParallelView::PVAbstractRangeAxisSliders::PVAbstractRangeAxisSliders
 *****************************************************************************/

PVParallelView::PVAbstractRangeAxisSliders::PVAbstractRangeAxisSliders(QGraphicsItem *parent,
                                                                       PVSlidersManager_p sm_p,
                                                                       PVParallelView::PVSlidersGroup *group,
                                                                       const char *text) :
	PVAbstractAxisSliders(parent, sm_p, group, text)
{
}

/*****************************************************************************
 * PVParallelView::PVAbstractRangeAxisSliders::paint
 *****************************************************************************/

void PVParallelView::PVAbstractRangeAxisSliders::paint(QPainter *painter,
                                                       const QStyleOptionGraphicsItem *,
                                                       QWidget *)
{
	if (is_moving()) {
		painter->save();

		painter->setCompositionMode(QPainter::RasterOp_SourceXorDestination);

		QPen new_pen(Qt::white);
		new_pen.setWidth(0);
		painter->setPen(new_pen);
		qreal vmin = _sl_min->pos().y();
		qreal vmax = _sl_max->pos().y();
		painter->drawLine(QPointF(0., vmin), QPointF(0., vmax));

		_text->setPos(0, vmin);
		_text->show();

		painter->restore();
	} else {
		_text->hide();
	}
}
