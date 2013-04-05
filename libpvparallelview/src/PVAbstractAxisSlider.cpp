
#include <pvkernel/core/PVLogger.h>

#include <pvparallelview/PVAbstractAxisSlider.h>
#include <pvparallelview/PVAbstractAxisSliders.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVSlidersGroup.h>

#include <QPainter>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QMenu>

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::PVAbstractAxisSlider
 *****************************************************************************/

PVParallelView::PVAbstractAxisSlider::PVAbstractAxisSlider(int64_t omin, int64_t omax, int64_t o,
                                                           PVAxisSliderOrientation orientation) :
	_offset_min(omin), _offset_max(omax), _offset(o),
	_orientation(orientation), _moving(false), _is_hover(false),
	_removable(true)
{
	setAcceptHoverEvents(true); // This is needed to enable hover events

	setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::~PVAbstractAxisSlider
 *****************************************************************************/

PVParallelView::PVAbstractAxisSlider::~PVAbstractAxisSlider()
{
	QGraphicsScene *s = scene();

	if (s != 0) {
		s->removeItem(this);
	}
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::set_value
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::set_value(int64_t v)
{
	_offset = PVCore::clamp(v, _offset_min, _offset_max);

	double f = _owner->get_sliders_group()->get_axis_scale();
	double vd = PVCore::clamp<int64_t>(_offset * f, _offset_min * f, _offset_max * f);
	vd /= (double)BUCKET_ELT_COUNT;

	setPos(0., vd);
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::hoverenterEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::hoverEnterEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	//PVLOG_INFO("PVAbstractAxisSlider::hoverEnterEvent\n");
	_is_hover = true;
	group()->update();
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::hoverMoveEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::hoverMoveEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	//PVLOG_INFO("PVAbstractAxisSlider::hoverMoveEvent\n");
	_is_hover = true;
	group()->update();
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::hoverLeaveEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::hoverLeaveEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	//PVLOG_INFO("PVAbstractAxisSlider::hoverLeaveEvent\n");
	_is_hover = false;
	group()->update();
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		_move_offset = pos().y() - event->scenePos().y();
		_moving = true;

		event->accept();
	}
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		emit slider_moved();
		_moving = false;
		event->accept();
	}
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->buttons() == Qt::LeftButton) {
		double sy = ((double)BUCKET_ELT_COUNT * (event->scenePos().y() + _move_offset)) / _owner->get_sliders_group()->get_axis_scale();
		set_value(sy);

		group()->update();
		event->accept();
	}
}


/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::contextMenuEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{
	if (_removable) {
		QMenu menu;

		QAction *rem = menu.addAction("Remove cursors");
		connect(rem, SIGNAL(triggered()), _owner, SLOT(remove_from_axis()));

		if (menu.exec(event->screenPos()) != nullptr) {
			event->accept();
		}
	}
}

void PVParallelView::PVAbstractAxisSlider::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
#ifdef PICVIZ_DEVELOPER_MODE
	if (common::show_bboxes()) {
		painter->save();
		painter->setPen(QColor(0xFF, 0, 0));
		painter->setBrush(QColor(0xFF, 0, 0, 128));
		const QRectF br = boundingRect();
		painter->drawRect(br);
		painter->restore();
	}
#endif
}
