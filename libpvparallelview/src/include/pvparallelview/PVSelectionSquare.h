/**
 * \file PVSelectionSquare.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVPARALLELVIEW_PVSELECTIONSQUARE_H__
#define __PVPARALLELVIEW_PVSELECTIONSQUARE_H__

#include <QObject>

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

namespace __impl {

struct PVSelectionBarycenter
{
	PVSelectionBarycenter()
	{
		clear();
	}

	PVZoneID zone_id1;
	PVZoneID zone_id2;
	double factor1;
	double factor2;

	void clear()
	{
		zone_id1 = PVZONEID_INVALID;
		zone_id2 = PVZONEID_INVALID;
		factor1 = 0.0;
		factor2 = 0.0;
	}
};

}

class PVSelectionSquare : public QObject
{
	Q_OBJECT;

public:
	PVSelectionSquare(PVFullParallelScene* s);

public:
	void begin(int x, int y);
	void end(int x, int y, bool now = false);
	void update_position();
	void clear();
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

private:
	void move_by(qreal hratio, qreal vratio);
	void grow_by(qreal hratio, qreal vratio);

public slots:
	void commit();

private:
	void store();

	PVFullParallelScene* scene_parent();
	PVFullParallelScene const* scene_parent() const;

	PVLinesView& get_lines_view();
	PVLinesView const& get_lines_view() const;

	Picviz::PVView& lib_view();
	Picviz::PVView const& lib_view() const;

private:
	PVSelectionSquareGraphicsItem* _selection_graphics_item;
	__impl::PVSelectionBarycenter _selection_barycenter;
	QPointF _selection_graphics_item_pos;
	bool _mouse = true;
};

}

#endif // __PVPARALLELVIEW_PVSELECTIONSQUARE_H__