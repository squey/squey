/**
 * \file PVSelectionSquare.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVPARALLELVIEW_PVSELECTIONSQUARE_H__
#define __PVPARALLELVIEW_PVSELECTIONSQUARE_H__

#include <QObject>
#include <QPointF>

#include <iostream>

#include <pvparallelview/PVSelectionSquareGraphicsItem.h>
#include <pvparallelview/PVSelectionGenerator.h>

constexpr qreal MOVE_STEP_PX = 1;
constexpr qreal GROW_STEP_RATIO = 1.2;

namespace Picviz {
class PVView;
}

namespace PVParallelView {

class PVFullParallelScene;

class PVSelectionSquare : public QObject
{
	Q_OBJECT;

public:
	PVSelectionSquare(Picviz::PVView& view, QGraphicsScene* s);
	virtual ~PVSelectionSquare() {};

public:
	void begin(qreal x, qreal y);
	void end(qreal x, qreal y, bool use_selection_modifiers = true, bool now = false);
	virtual void clear();

	void update_rect_no_commit(const QRectF& r);
	QRectF get_rect();

public:
	void move_left_by_step() { move_by(-MOVE_STEP_PX, 0); }
	void move_right_by_step() { move_by(MOVE_STEP_PX, 0); }
	void move_up_by_step() { move_by(0, -MOVE_STEP_PX); }
	void move_down_by_step() { move_by(0, MOVE_STEP_PX); }

	void move_left_by_width() { move_by(-_selection_graphics_item->rect().width(), 0); }
	void move_right_by_width() { move_by(_selection_graphics_item->rect().width(), 0); }
	void move_up_by_height() { move_by(0, -_selection_graphics_item->rect().height()); }
	void move_down_by_height() { move_by(0, _selection_graphics_item->rect().height()); }

	void grow_horizontally() { grow_by(GROW_STEP_RATIO, 1); }
	void shrink_horizontally() { grow_by(1/GROW_STEP_RATIO, 1); }
	void grow_vertically() { grow_by(1, 1/GROW_STEP_RATIO); };
	void shrink_vertically() { grow_by(1, GROW_STEP_RATIO); };

protected slots:
	virtual void commit(bool use_selection_modifiers) = 0;

private:
	void move_by(qreal hratio, qreal vratio);
	void grow_by(qreal hratio, qreal vratio);

protected:
	Picviz::PVView& _view;
	PVSelectionSquareGraphicsItem* _selection_graphics_item;
	QPointF _selection_graphics_item_pos;
};

}

#endif // __PVPARALLELVIEW_PVSELECTIONSQUARE_H__
