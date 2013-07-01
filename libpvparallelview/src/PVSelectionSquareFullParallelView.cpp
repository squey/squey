/**
 * \file PVSelectionSquareFullParallelView.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVSelectionSquareFullParallelView.h>

#include <pvparallelview/PVFullParallelScene.h>

PVParallelView::PVSelectionSquareFullParallelView::PVSelectionSquareFullParallelView(QGraphicsScene* s) :
	PVSelectionSquare(s)
{
}

void PVParallelView::PVSelectionSquareFullParallelView::commit(bool use_selection_modifiers)
{
	Picviz::PVView& view = lib_view();
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
		PVSelectionGenerator::compute_selection_from_parallel_view_rect(get_lines_view(), z, r, lib_view().get_volatile_selection());
	}

	store();

	PVSelectionGenerator::process_selection(view.shared_from_this(), use_selection_modifiers);
}

void PVParallelView::PVSelectionSquareFullParallelView::clear()
{
	PVSelectionSquare::clear();
	_selection_barycenter.clear();
}

void PVParallelView::PVSelectionSquareFullParallelView::store()
{
	PVZoneID& zone_id1 = _selection_barycenter.zone_id1;
	PVZoneID& zone_id2 = _selection_barycenter.zone_id2;
	double& factor1 = _selection_barycenter.factor1;
	double& factor2 = _selection_barycenter.factor2;

	double abs_left = _selection_graphics_item->rect().topLeft().x();
	double abs_right = _selection_graphics_item->rect().bottomRight().x();

	zone_id1 = get_lines_view().get_zone_from_scene_pos(abs_left);
	double z1_width = get_lines_view().get_zone_width(zone_id1);
	double alpha = scene_parent()->map_to_axis(zone_id1, QPointF(abs_left, 0)).x();
	factor1 = (double) alpha / z1_width;

	zone_id2 = get_lines_view().get_zone_from_scene_pos(abs_right);
	double z2_width = get_lines_view().get_zone_width(zone_id2);
	double beta = scene_parent()->map_to_axis(zone_id2, QPointF(abs_right, 0)).x();
	factor2 = (double) beta / z2_width;
}

void PVParallelView::PVSelectionSquareFullParallelView::update_position()
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

	double new_left = get_lines_view().get_left_border_position_of_zone_in_scene(zone_id1) + (double) get_lines_view().get_zone_width(zone_id1) * factor1;
	double new_right = get_lines_view().get_left_border_position_of_zone_in_scene(zone_id2) + (double) get_lines_view().get_zone_width(zone_id2) * factor2;
	double abs_top = _selection_graphics_item->rect().topLeft().y();
	double abs_bottom = _selection_graphics_item->rect().bottomRight().y();

	_selection_graphics_item->setRect(QRectF(QPointF(new_left, abs_top), QPointF(new_right, abs_bottom)));
}

PVParallelView::PVFullParallelScene* PVParallelView::PVSelectionSquareFullParallelView::scene_parent() { return static_cast<PVParallelView::PVFullParallelScene*>(parent()); }
PVParallelView::PVFullParallelScene const* PVParallelView::PVSelectionSquareFullParallelView::scene_parent() const { return static_cast<PVParallelView::PVFullParallelScene const*>(parent()); }

PVParallelView::PVLinesView& PVParallelView::PVSelectionSquareFullParallelView::get_lines_view() { return scene_parent()->get_lines_view(); }
PVParallelView::PVLinesView const& PVParallelView::PVSelectionSquareFullParallelView::get_lines_view() const { return scene_parent()->get_lines_view(); }

Picviz::PVView& PVParallelView::PVSelectionSquareFullParallelView::lib_view() { return scene_parent()->lib_view(); }
