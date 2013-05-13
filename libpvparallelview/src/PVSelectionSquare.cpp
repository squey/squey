/**
 * \file PVSelectionSquare.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVSelectionSquare.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVLinesView.h>

PVParallelView::PVSelectionSquare::PVSelectionSquare(QGraphicsScene* s):
	QObject(static_cast<QObject*>(s)),
	_selection_graphics_item(new PVSelectionSquareGraphicsItem())
{
	// PVselectionSquare will belong to the parent QGraphicsScene, thus will be
	// deleted by the scene when it will be deleted.
	// The same goes for _selection_graphics_item.
	s->addItem(_selection_graphics_item);
	_selection_graphics_item->hide();

	connect(_selection_graphics_item, SIGNAL(commit_volatile_selection(bool)), this, SLOT(commit(bool)));
}

void PVParallelView::PVSelectionSquare::begin(qreal x, qreal y)
{
	_selection_graphics_item_pos = QPointF(x, y);
	_selection_graphics_item->show();
}

void PVParallelView::PVSelectionSquare::end(qreal x, qreal y, bool use_selection_modifiers /* = true */, bool now /*= false */)
{
	Picviz::PVView& view = lib_view();

	QPointF p(x, y);

	if (_selection_graphics_item_pos != p) {
		QPointF top_left(qMin(_selection_graphics_item_pos.x(), p.x()),
		                 qMin(_selection_graphics_item_pos.y(), p.y()));
		QPointF bottom_right(qMax(_selection_graphics_item_pos.x(), p.x()),
		                     qMax(_selection_graphics_item_pos.y(), p.y()));
		_selection_graphics_item->update_rect(QRectF(top_left, bottom_right),
		                                      use_selection_modifiers, now);
	}
	else {
		clear();
		view.get_volatile_selection().select_none();
		PVSelectionGenerator::process_selection(view.shared_from_this(), false);
		//scene_parent()->process_selection(false);
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

	begin(x-hoffset, y-voffset);
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

QGraphicsScene* PVParallelView::PVSelectionSquare::scene() const
{
	return _selection_graphics_item->scene();
}
