
#include <pvparallelview/PVSelectionRectangle.h>
#include <pvparallelview/PVSelectionRectangleItem.h>

#include <QGraphicsScene>
#include <QActionGroup>
#include <QSignalMapper>
#include <QToolButton>
#include <QToolBar>
#include <QAction>

const QColor PVParallelView::PVSelectionRectangle::rectangle_color     = Qt::red;
const QColor PVParallelView::PVSelectionRectangle::handle_color        = QColor(255, 127, 36);
const int    PVParallelView::PVSelectionRectangle::handle_transparency = 50;
const int    PVParallelView::PVSelectionRectangle::delay_msec          = 300;

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::PVSelectionRectangle
 *****************************************************************************/

PVParallelView::PVSelectionRectangle::PVSelectionRectangle(QGraphicsScene* scene) :
	QObject(static_cast<QObject*>(scene)),
	_use_selection_modifiers(true)
{
	_rect = new PVParallelView::PVSelectionRectangleItem();
	scene->addItem(_rect);
	_rect->clear();
	_rect->set_pen_color(PVSelectionRectangle::rectangle_color);

	connect(_rect, SIGNAL(geometry_has_changed(const QRectF&, const QRectF&)),
	        this, SLOT(start_timer()));

	QColor hc = PVSelectionRectangle::handle_color;
	_rect->set_handles_pen_color(hc);
	hc.setAlpha(PVSelectionRectangle::handle_transparency);
	_rect->set_handles_brush_color(hc);

	_timer = new QTimer(this);
	_timer->setSingleShot(true);

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

void PVParallelView::PVSelectionRectangle::end(const QPointF& p,
                                               bool use_sel_modifiers,
                                               bool now)
{
	_use_selection_modifiers = use_sel_modifiers;
	_rect->end(p);
	if (now) {
		timeout();
	} else {
		start_timer();
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::scene
 *****************************************************************************/

QGraphicsScene* PVParallelView::PVSelectionRectangle::scene() const
{
	return _rect->scene();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::add_selection_mode_selector
 *****************************************************************************/

QToolButton* PVParallelView::PVSelectionRectangle::add_selection_mode_selector(QWidget *view,
                                                                              QToolBar *toolbar,
                                                                              QSignalMapper* signal_mapper)
{
	toolbar->setIconSize(QSize(17, 17));

	QToolButton* selection_mode = new QToolButton(toolbar);
	selection_mode->setPopupMode(QToolButton::InstantPopup);
	selection_mode->setIcon(QIcon(":/selection-rectangle"));
	selection_mode->setToolTip(tr("Selection mode"));

	// Rectangle selection
	QAction* r_sel = new QAction("Rectangle", toolbar);
	r_sel->setIcon(QIcon(":/selection-rectangle"));
	r_sel->setShortcut(Qt::Key_R);
	selection_mode->addAction(r_sel);
	view->addAction(r_sel);
	signal_mapper->setMapping(r_sel, SelectionMode::RECTANGLE);
	QObject::connect(r_sel, SIGNAL(triggered(bool)), signal_mapper, SLOT(map()));

	// Horizontal selection
	QAction* h_sel = new QAction("Horizontal", toolbar);
	h_sel->setIcon(QIcon(":/selection-horizontal"));
	h_sel->setShortcut(Qt::Key_H);
	selection_mode->addAction(h_sel);
	view->addAction(h_sel);
	signal_mapper->setMapping(h_sel, SelectionMode::HORIZONTAL);
	QObject::connect(h_sel, SIGNAL(triggered(bool)), signal_mapper, SLOT(map()));

	// Vertical selection
	QAction* v_sel = new QAction("Vertical", toolbar);
	v_sel->setIcon(QIcon(":/selection-vertical"));
	v_sel->setShortcut(Qt::Key_V);
	selection_mode->addAction(v_sel);
	view->addAction(v_sel);
	signal_mapper->setMapping(v_sel, SelectionMode::VERTICAL);
	QObject::connect(v_sel, SIGNAL(triggered(bool)), signal_mapper, SLOT(map()));

	toolbar->addWidget(selection_mode);

	return selection_mode;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::update_selection_mode_selector
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::update_selection_mode_selector(QToolButton* button,
                                                                          int mode)
{
	QAction *action = nullptr;

	try {
		// QList::at has assert in DEBUG mode...
		action = button->actions().at(mode);
	} catch(...) {
	}

	if (action != nullptr) {
		button->setIcon(action->icon());
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::start_timer
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::start_timer()
{
	_rect->set_pen_color(handle_color);
	_timer->start(PVSelectionRectangle::delay_msec);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::timeout
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::timeout()
{
	_rect->set_pen_color(PVSelectionRectangle::rectangle_color);
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

