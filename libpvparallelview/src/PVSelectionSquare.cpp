/**
 * \file PVSelectionSquare.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVSelectionSquare.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVLinesView.h>

PVParallelView::PVSelectionSquare::PVSelectionSquare(Picviz::PVView& view, QGraphicsScene* s) :
	QObject((QObject*)s),
	_view(view),
	_selection_graphics_item(new PVSelectionSquareGraphicsItem((QGraphicsScene*)s))
{
	_selection_graphics_item->hide();
	connect(_selection_graphics_item, SIGNAL(commit_volatile_selection(bool)), this, SLOT(commit(bool)));
}

void PVParallelView::PVSelectionSquare::begin(int x, int y)
{
	_selection_graphics_item_pos = QPointF(qreal(x), qreal(y));
	_selection_graphics_item->show();
}

void PVParallelView::PVSelectionSquare::end(int x, int y, bool use_selection_modifiers /* = true */, bool now /*= false */)
{
	if (_selection_graphics_item_pos != QPointF(x, y)) {
		qreal xF = qreal(x);
		qreal yF = qreal(y);
		QPointF top_left(qMin(_selection_graphics_item_pos.x(), xF), qMin(_selection_graphics_item_pos.y(), yF));
		QPointF bottom_right(qMax(_selection_graphics_item_pos.x(), xF), qMax(_selection_graphics_item_pos.y(), yF));
		_selection_graphics_item->update_rect(QRectF(top_left, bottom_right), use_selection_modifiers, now);
	}
	else {
		clear();
		_view.get_volatile_selection().select_none();//lib_view().get_volatile_selection().select_none();
		PVSelectionGenerator::process_selection(_view.shared_from_this(), false);//scene_parent()->process_selection(false);
	}
}

void PVParallelView::PVSelectionSquare::clear()
{
	_selection_graphics_item->clear_rect();
}

void PVParallelView::PVSelectionSquare::move_by(qreal hstep, qreal vstep)
{
	qreal width = _selection_graphics_item->rect().width();
	qreal height = _selection_graphics_item->rect().height();
	qreal x = _selection_graphics_item->rect().x();
	qreal y = _selection_graphics_item->rect().y();

	begin(x+hstep, y+vstep);
	end(x+hstep+width, y+vstep+height, false);
}

void PVParallelView::PVSelectionSquare::grow_by(qreal hratio, qreal vratio)
{
	qreal width = std::max((qreal)1, _selection_graphics_item->rect().width());
	qreal height = std::max((qreal)1,_selection_graphics_item->rect().height());
	qreal x = _selection_graphics_item->rect().x();
	qreal y = _selection_graphics_item->rect().y();

	qreal hoffset = (width-width*hratio);
	qreal voffset = (height-height*vratio);

	begin(x-hoffset/2, y-voffset/2);
	end(x+hoffset+width, y+voffset+height, false);
}

QRectF PVParallelView::PVSelectionSquare::get_rect()
{
	return _selection_graphics_item->rect();
}

void PVParallelView::PVSelectionSquare::update_rect_no_commit(const QRectF& r)
{
	_selection_graphics_item->update_rect_no_commit(r);
}
