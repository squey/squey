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

class PVSelectionSquare : public QObject
{
	Q_OBJECT;

public:
	using EMode = PVSelectionSquareGraphicsItem::EMode;

public:
	PVSelectionSquare(QGraphicsScene* scene);
	virtual ~PVSelectionSquare() {};

public:
	void begin(qreal x, qreal y);
	void end(qreal x, qreal y, bool use_selection_modifiers = true, bool now = false);
	virtual void clear();

	void update_rect_no_commit(const QRectF& r);
	QRectF get_rect();

	inline void hide() { _selection_graphics_item->hide(); }

public:
	void move_left_by_step() { move_by(-MOVE_STEP_PX, 0); }
	void move_right_by_step() { move_by(MOVE_STEP_PX, 0); }
	void move_horizontally_by_step(bool left) { move_by((left ? -1 : 1 ) * MOVE_STEP_PX, 0); }

	void move_up_by_step() { move_by(0, -MOVE_STEP_PX); }
	void move_down_by_step() { move_by(0, MOVE_STEP_PX); }
	void move_vertically_by_step(bool up) { move_by(0, (up ? -1 : 1) * MOVE_STEP_PX); }

	void move_left_by_width() { move_by(-_selection_graphics_item->rect().width(), 0); }
	void move_right_by_width() { move_by(_selection_graphics_item->rect().width(), 0); }
	void move_horizontally_by_width(bool left) { move_by((left ? -1 : 1) * _selection_graphics_item->rect().width(), 0); }

	void move_up_by_height() { move_by(0, -_selection_graphics_item->rect().height()); }
	void move_down_by_height() { move_by(0, _selection_graphics_item->rect().height()); }
	void move_vertically_by_height(bool up) { move_by(0, (up ? -1 : 1 ) * _selection_graphics_item->rect().height()); }

	void grow_horizontally() { grow_by(GROW_STEP_RATIO, 1); }
	void shrink_horizontally() { grow_by(1/GROW_STEP_RATIO, 1); }
	void grow_vertically() { grow_by(1, 1/GROW_STEP_RATIO); }
	void shrink_vertically() { grow_by(1, GROW_STEP_RATIO); }

	EMode selection_mode() const
	{
		return _selection_graphics_item->selection_mode();
	}

	QGraphicsScene* scene() const;

public slots:
	void set_selection_mode(int selection_mode)
	{
		_selection_graphics_item->set_selection_mode(selection_mode);
	}

protected slots:
	virtual void commit(bool use_selection_modifiers) = 0;
	virtual Picviz::PVView& lib_view() = 0;

private:
	void move_by(qreal hratio, qreal vratio);
	void grow_by(qreal hratio, qreal vratio);

protected:
	PVSelectionSquareGraphicsItem* _selection_graphics_item;
	QPointF _selection_graphics_item_pos;
};

}

#endif // __PVPARALLELVIEW_PVSELECTIONSQUARE_H__
