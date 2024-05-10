//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvparallelview/PVSelectionRectangle.h>
#include <pvparallelview/PVSelectionRectangleItem.h>
#include <pvparallelview/PVSelectionHandleItem.h>

#include <pvkernel/widgets/PVModdedIcon.h>

#include <QGraphicsScene>
#include <QActionGroup>
#include <QSignalMapper>
#include <QToolButton>
#include <QToolBar>
#include <QAction>

const QColor PVParallelView::PVSelectionRectangle::rectangle_color = Qt::red;
const QColor PVParallelView::PVSelectionRectangle::handle_color = QColor(255, 127, 36);
const int PVParallelView::PVSelectionRectangle::handle_transparency = 50;
const int PVParallelView::PVSelectionRectangle::delay_msec = 300;

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::PVSelectionRectangle
 *****************************************************************************/

PVParallelView::PVSelectionRectangle::PVSelectionRectangle(QGraphicsScene* scene)
    : QObject(static_cast<QObject*>(scene)), _use_selection_modifiers(true)
{
	_rect = new PVParallelView::PVSelectionRectangleItem();
	scene->addItem(_rect);
	_rect->clear();
	_rect->set_pen_color(PVSelectionRectangle::rectangle_color);

	connect(_rect, &PVSelectionRectangleItem::geometry_has_changed, this,
	        &PVSelectionRectangle::start_timer);

	QColor hc = PVSelectionRectangle::handle_color;
	_rect->set_handles_pen_color(hc);
	hc.setAlpha(PVSelectionRectangle::handle_transparency);
	_rect->set_handles_brush_color(hc);

	_timer = new QTimer(this);
	_timer->setSingleShot(true);

	connect(_timer, &QTimer::timeout, this, &PVSelectionRectangle::timeout);

	/* as commit is a virtual method, the chaining
	 * timeout->commit_volatile_selection->commit is required
	 */
	connect(this, &PVSelectionRectangle::commit_volatile_selection, this,
	        &PVSelectionRectangle::commit);
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

void PVParallelView::PVSelectionRectangle::end(const QPointF& p, bool use_sel_modifiers, bool now)
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

QToolButton* PVParallelView::PVSelectionRectangle::add_selection_mode_selector(
    QWidget* view, QToolBar* toolbar, QSignalMapper* signal_mapper)
{
	toolbar->setIconSize(QSize(17, 17));

	auto selection_mode = new QToolButton(toolbar);
	selection_mode->setPopupMode(QToolButton::InstantPopup);
	selection_mode->setIcon(PVModdedIcon("selection-square"));
	selection_mode->setToolTip(tr("Selection mode"));

	// Rectangle selection
	auto* r_sel = new QAction("Rectangle", toolbar);
	r_sel->setIcon(PVModdedIcon("selection-square"));
	r_sel->setShortcut(Qt::Key_R);
	r_sel->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	selection_mode->addAction(r_sel);
	view->addAction(r_sel);
	signal_mapper->setMapping(r_sel, SelectionMode::RECTANGLE);
	QObject::connect(r_sel, &QAction::triggered, signal_mapper,
	                 static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));

	// Horizontal selection
	auto* h_sel = new QAction("Horizontal", toolbar);
	h_sel->setIcon(PVModdedIcon("selection-horizontal"));
	h_sel->setShortcut(Qt::Key_H);
	h_sel->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	selection_mode->addAction(h_sel);
	view->addAction(h_sel);
	signal_mapper->setMapping(h_sel, SelectionMode::HORIZONTAL);
	QObject::connect(h_sel, &QAction::triggered, signal_mapper,
	                 static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));

	// Vertical selection
	auto* v_sel = new QAction("Vertical", toolbar);
	v_sel->setIcon(PVModdedIcon("selection-vertical"));
	v_sel->setShortcut(Qt::Key_V);
	v_sel->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	selection_mode->addAction(v_sel);
	view->addAction(v_sel);
	signal_mapper->setMapping(v_sel, SelectionMode::VERTICAL);
	QObject::connect(v_sel, &QAction::triggered, signal_mapper,
	                 static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));

	toolbar->addWidget(selection_mode);

	return selection_mode;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::update_selection_mode_selector
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::update_selection_mode_selector(QToolButton* button,
                                                                          int mode)
{
	QAction* action = nullptr;

	try {
		// QList::at has assert in DEBUG mode...
		action = button->actions().at(mode);
	} catch (...) {
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
	_rect->set_pen_color(PVSelectionRectangle::handle_color);
	_rect->get_central_handle()->set_pen_color(PVSelectionRectangle::handle_color);

	_timer->start(PVSelectionRectangle::delay_msec);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::timeout
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::timeout()
{
	_rect->set_pen_color(PVSelectionRectangle::rectangle_color);
	_rect->get_central_handle()->set_pen_color(PVSelectionRectangle::rectangle_color);

	_timer->stop();

	Q_EMIT commit_volatile_selection(_use_selection_modifiers);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::move_by
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::move_by(qreal hstep, qreal vstep)
{
	const QRectF old_rect = get_rect();

	const qreal x = old_rect.x() + hstep;
	const qreal y = old_rect.y() + vstep;
	const qreal width = old_rect.width();
	const qreal height = old_rect.height();

	begin(QPointF(x, y));
	end(QPointF(x + width, y + height), false, true);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangle::grow_by
 *****************************************************************************/

void PVParallelView::PVSelectionRectangle::grow_by(qreal hratio, qreal vratio)
{
	QRectF old_rect = get_rect();

	qreal width = std::max((qreal)1, old_rect.width());
	qreal height = std::max((qreal)1, old_rect.height());
	qreal x = old_rect.x();
	qreal y = old_rect.y();

	qreal hoffset = (width - width * hratio);
	qreal voffset = (height - height * vratio);

	begin(QPointF(x - hoffset, y - voffset));
	end(QPointF(x + hoffset + width, y + voffset + height), false, true);
}
