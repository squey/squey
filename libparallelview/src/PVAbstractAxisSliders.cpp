
#include <pvparallelview/PVAbstractAxisSliders.h>

#include <QGraphicsSimpleTextItem>
#include <QPainter>

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSliders::PVAbstractAxisSliders
 *****************************************************************************/

PVParallelView::PVAbstractAxisSliders::PVAbstractAxisSliders(QGraphicsItem *parent,
                                                             PVSlidersManager_p sm_p,
                                                             PVSlidersGroup *group,
                                                             const char *text) :
	QObject(nullptr), QGraphicsItemGroup(parent),
	_sliders_manager_p(sm_p),
	_group(group)
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
