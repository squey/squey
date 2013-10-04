
#include <pvparallelview/PVSelectionRectangle.h>
#include <pvparallelview/PVSelectionRectangleItem.h>

#include <QGraphicsScene>

const QColor PVParallelView::PVSelectionRectangle::rect_color(Qt::red);
const QColor PVParallelView::PVSelectionRectangle::handle_color(255, 127, 36);

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::PVSelectionRectangle
 *****************************************************************************/

PVParallelView::PVSelectionRectangle::PVSelectionRectangle(QGraphicsScene* scene) :
	QObject(static_cast<QObject*>(scene)),
	_use_selection_modifiers(true),
	_sel_mode(RECTANGLE)
{
	_rect = new PVParallelView::PVSelectionRectangleItem();
	scene->addItem(_rect);
	_rect->hide();
	_rect->set_pen_color(rect_color);

	connect(_rect, SIGNAL(geometry_has_changed(const QRectF&, const QRectF&)),
	        this, SLOT(start_timer()));

	QColor hc = handle_color;
	_rect->set_handles_pen_color(hc);
	hc.setAlpha(50);
	_rect->set_handles_brush_color(hc);

	_timer = new QTimer(this);
	connect(_timer, SIGNAL(timeout()),
	        this, SLOT(timeout()));

	/* as commit is a virtual method, the chaining
	 * timeout->commit_volatile_selection->commit is required
	 */
	connect(this, SIGNAL(commit_volatile_selection(bool)),
	        this, SLOT(commit(bool)));
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::clear
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::clear()
{
	_rect->clear();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::begin
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::begin(const QPointF& p)
{
	_rect->begin(p);
	start_timer();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::step
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::step(const QPointF& p)
{
	_rect->step(p);
	start_timer();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::end
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::end(const QPointF& p)
{
	_rect->end(p);
	start_timer();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::scene
 *****************************************************************************/

QGraphicsScene* PVParallelView::PVSelectionRectangle::scene() const
{
	return _rect->scene();
}
/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::start_timer
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::start_timer()
{
	_rect->set_pen_color(handle_color);
	_timer->start(300);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::timeout
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::timeout()
{
	_rect->set_pen_color(rect_color);
	_timer->stop();
	emit commit_volatile_selection(_use_selection_modifiers);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::move_by
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::move_by(qreal hstep, qreal vstep)
{
	const QRectF old_rect = get_rect();
	_rect->set_rect(QRectF(old_rect.x() + hstep,
	                       old_rect.y() + vstep,
	                       old_rect.width(),
	                       old_rect.height()));

	start_timer();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::grow_by
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::grow_by(qreal hratio, qreal vratio)
{
	QRectF old_rect = get_rect();
	const qreal width = std::max((qreal)1, old_rect.width());
	const qreal height = std::max((qreal)1, old_rect.height());

	const qreal x = old_rect.x() + width * .5;
	const qreal y = old_rect.y() + height * .5;

	const qreal nwidth  = width / hratio;
	const qreal nheight = height / vratio;

	_rect->set_rect(QRectF(x - .5 * nwidth,
	                       y - .5 * nheight,
	                       nwidth,
	                       nheight));
	start_timer();
}

