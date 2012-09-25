
#include <pvparallelview/PVAbstractRangeAxisSliders.h>

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
		painter->drawLine(0, _sl_min->value(), 0, _sl_max->value());

		_text->setPos(0, _sl_min->value());
		_text->show();

		painter->restore();
	} else {
		_text->hide();
	}
}
