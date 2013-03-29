/**
 * \file PVSelectionSquare.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVSelectionSquare.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVLinesView.h>

PVParallelView::PVSelectionSquare::PVSelectionSquare(PVFullParallelScene* s) :
	QObject((QObject*)s),
	_selection_graphics_item(new PVSelectionSquareGraphicsItem((QGraphicsScene*)s))
{
	_selection_graphics_item->hide();
	connect(_selection_graphics_item, SIGNAL(commit_volatile_selection()), this, SLOT(commit()));
}

void PVParallelView::PVSelectionSquare::begin(int x, int y)
{
	_selection_graphics_item_pos = QPointF(qreal(x), qreal(y));
	_selection_graphics_item->show();
}

void PVParallelView::PVSelectionSquare::end(int x, int y, bool now /*= false */)
{
	qreal xF = qreal(x);
	qreal yF = qreal(y);
	QPointF top_left(qMin(_selection_graphics_item_pos.x(), xF), qMin(_selection_graphics_item_pos.y(), yF));
	QPointF bottom_right(qMax(_selection_graphics_item_pos.x(), xF), qMax(_selection_graphics_item_pos.y(), yF));
	_selection_graphics_item->update_rect(QRectF(top_left, bottom_right), now);
}

void PVParallelView::PVSelectionSquare::clear()
{
	_selection_barycenter.clear();
	_selection_graphics_item->clear_rect();
}

void PVParallelView::PVSelectionSquare::commit()
{
	_selection_graphics_item->finished();
	QRectF srect = _selection_graphics_item->rect();
	// Too much on the left dude!
	if (srect.x() + srect.width() <= 0) {
		return;
	}

	// Too much on the right, stop drinking!
	const int32_t pos_end = scene_parent()->pos_last_axis();
	if (srect.x() >= pos_end) {
		return;
	}

	const PVZoneID zone_id_start = get_lines_view().get_zone_from_scene_pos(srect.x());
	const PVZoneID zone_id_end = get_lines_view().get_zone_from_scene_pos(srect.x() + srect.width());

	lib_view().get_volatile_selection().select_none();
	for (PVZoneID z = zone_id_start; z <= zone_id_end; z++) {
		QRect r = scene_parent()->map_to_axis(z, srect);
		r.setX(picviz_max(0, r.x()));
		r.setRight(picviz_min(pos_end-1, r.right()));
		PVSelectionGenerator::compute_selection_from_rect(get_lines_view(), z, r, lib_view().get_volatile_selection());
	}

	store();

	if (_mouse) {
		scene_parent()->process_mouse_selection();
	}
	else {
		scene_parent()->process_key_selection();
	}
}

void PVParallelView::PVSelectionSquare::store()
{
	PVZoneID& zone_id1 = _selection_barycenter.zone_id1;
	PVZoneID& zone_id2 = _selection_barycenter.zone_id2;
	double& factor1 = _selection_barycenter.factor1;
	double& factor2 = _selection_barycenter.factor2;

	uint32_t abs_left = _selection_graphics_item->rect().topLeft().x();
	uint32_t abs_right = _selection_graphics_item->rect().bottomRight().x();

	zone_id1 = get_lines_view().get_zone_from_scene_pos(abs_left);
	uint32_t z1_width = get_lines_view().get_zone_width(zone_id1);
	uint32_t alpha = scene_parent()->map_to_axis(zone_id1, QPointF(abs_left, 0)).x();
	factor1 = (double) alpha / z1_width;

	zone_id2 = get_lines_view().get_zone_from_scene_pos(abs_right);
	uint32_t z2_width = get_lines_view().get_zone_width(zone_id2);
	uint32_t beta = scene_parent()->map_to_axis(zone_id2, QPointF(abs_right, 0)).x();
	factor2 = (double) beta / z2_width;
}

void PVParallelView::PVSelectionSquare::update_position()
{
	PVZoneID zone_id1 = _selection_barycenter.zone_id1;
	PVZoneID zone_id2 = _selection_barycenter.zone_id2;
	if ((zone_id1 == PVZONEID_INVALID) || (zone_id2 == PVZONEID_INVALID)) {
		return;
	}

	if (zone_id1 >= get_lines_view().get_number_of_managed_zones() ||
		zone_id2 >= get_lines_view().get_number_of_managed_zones()) {
		clear();
		return;
	}

	double factor1 = _selection_barycenter.factor1;
	double factor2 = _selection_barycenter.factor2;

	uint32_t new_left = get_lines_view().get_left_border_position_of_zone_in_scene(zone_id1) + (double) get_lines_view().get_zone_width(zone_id1) * factor1;
	uint32_t new_right = get_lines_view().get_left_border_position_of_zone_in_scene(zone_id2) + (double) get_lines_view().get_zone_width(zone_id2) * factor2;
	uint32_t abs_top = _selection_graphics_item->rect().topLeft().y();
	uint32_t abs_bottom = _selection_graphics_item->rect().bottomRight().y();

	_selection_graphics_item->setRect(QRectF(QPointF(new_left, abs_top), QPointF(new_right, abs_bottom)));
}

void PVParallelView::PVSelectionSquare::move_by(qreal hstep, qreal vstep)
{
	_mouse = false;

	qreal width = _selection_graphics_item->rect().width();
	qreal height = _selection_graphics_item->rect().height();
	qreal x = _selection_graphics_item->rect().x();
	qreal y = _selection_graphics_item->rect().y();

	begin(x+hstep, y+vstep);
	end(x+hstep+width, y+vstep+height, true);

	_mouse = true;
}

void PVParallelView::PVSelectionSquare::grow_by(qreal hratio, qreal vratio)
{
	_mouse = false;

	qreal width = _selection_graphics_item->rect().width();
	qreal height = _selection_graphics_item->rect().height();
	qreal x = _selection_graphics_item->rect().x();
	qreal y = _selection_graphics_item->rect().y();

	qreal hoffset = (width-width*hratio);
	qreal voffset = (height-height*vratio);

	begin(x-hoffset/2, y-voffset/2);
	end(x+hoffset+width, y+voffset+height, true);

	_mouse = true;
}

QRectF PVParallelView::PVSelectionSquare::get_rect()
{
	return _selection_graphics_item->rect();
}

void PVParallelView::PVSelectionSquare::update_rect_no_commit(const QRectF& r)
{
	_selection_graphics_item->update_rect_no_commit(r);
}

PVParallelView::PVFullParallelScene* PVParallelView::PVSelectionSquare::scene_parent() { return static_cast<PVParallelView::PVFullParallelScene*>(parent()); }
PVParallelView::PVFullParallelScene const* PVParallelView::PVSelectionSquare::scene_parent() const { return static_cast<PVParallelView::PVFullParallelScene const*>(parent()); }

PVParallelView::PVLinesView& PVParallelView::PVSelectionSquare::get_lines_view() { return scene_parent()->get_lines_view(); }
PVParallelView::PVLinesView const& PVParallelView::PVSelectionSquare::get_lines_view() const { return scene_parent()->get_lines_view(); }

Picviz::PVView& PVParallelView::PVSelectionSquare::lib_view() { return scene_parent()->lib_view(); }
Picviz::PVView const& PVParallelView::PVSelectionSquare::lib_view() const { return scene_parent()->lib_view(); }
