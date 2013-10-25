/**
 * \file PVFullParallelViewSelectionRectangle.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVFullParallelViewSelectionRectangle.h>

#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVSelectionGenerator.h>

#include <iostream>

#define print_r(R) __print_rect(#R, R)
#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << ", "
	          << r.width() << " " << r.height()
	          << std::endl;
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::PVFullParallelViewSelectionRectangle
 *****************************************************************************/

PVParallelView::PVFullParallelViewSelectionRectangle::PVFullParallelViewSelectionRectangle(PVFullParallelScene* fps) :
	PVSelectionRectangle(fps),
	_fps(fps)
{
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::clear
 *****************************************************************************/

void PVParallelView::PVFullParallelViewSelectionRectangle::clear()
{
	PVSelectionRectangle::clear();
	_barycenter.clear();
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::updsate_position
 *****************************************************************************/

void PVParallelView::PVFullParallelViewSelectionRectangle::update_position()
{
	PVZoneID zone_id1 = _barycenter.zone_id1;
	PVZoneID zone_id2 = _barycenter.zone_id2;

	if ((zone_id1 == PVZONEID_INVALID) || (zone_id2 == PVZONEID_INVALID)) {
		return;
	}

	if (zone_id1 >= get_lines_view().get_number_of_managed_zones()
	    || zone_id2 >= get_lines_view().get_number_of_managed_zones()) {
		clear();
		return;
	}

	double factor1 = _barycenter.factor1;
	double factor2 = _barycenter.factor2;

	double new_left = get_lines_view().get_left_border_position_of_zone_in_scene(zone_id1) + (double) get_lines_view().get_zone_width(zone_id1) * factor1;
	double new_right = get_lines_view().get_left_border_position_of_zone_in_scene(zone_id2) + (double) get_lines_view().get_zone_width(zone_id2) * factor2;
	double abs_top = get_rect().top();
	double abs_bottom = get_rect().bottom();

	set_rect(QRectF(QPointF(new_left, abs_top),
	                QPointF(new_right, abs_bottom)));
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::commit
 *****************************************************************************/

void PVParallelView::PVFullParallelViewSelectionRectangle::commit(bool use_selection_modifiers)
{
	Picviz::PVView& view = lib_view();
	QRectF srect = get_rect();

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

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::lib_view
 *****************************************************************************/

Picviz::PVView& PVParallelView::PVFullParallelViewSelectionRectangle::lib_view()
{
	return scene_parent()->lib_view();
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::store
 *****************************************************************************/

void PVParallelView::PVFullParallelViewSelectionRectangle::store()
{
	PVZoneID& zone_id1 = _barycenter.zone_id1;
	PVZoneID& zone_id2 = _barycenter.zone_id2;
	double& factor1 = _barycenter.factor1;
	double& factor2 = _barycenter.factor2;

	const double abs_left = get_rect().left();
	const double abs_right = get_rect().right();

	zone_id1 = get_lines_view().get_zone_from_scene_pos(abs_left);
	const double z1_width = get_lines_view().get_zone_width(zone_id1);
	const double alpha = scene_parent()->map_to_axis(zone_id1, QPointF(abs_left, 0)).x();
	factor1 = (double) alpha / z1_width;

	zone_id2 = get_lines_view().get_zone_from_scene_pos(abs_right);
	const double z2_width = get_lines_view().get_zone_width(zone_id2);
	const double beta = scene_parent()->map_to_axis(zone_id2, QPointF(abs_right, 0)).x();
	factor2 = (double) beta / z2_width;
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::scene_parent
 *****************************************************************************/

PVParallelView::PVFullParallelScene* PVParallelView::PVFullParallelViewSelectionRectangle::scene_parent()
{
	return static_cast<PVParallelView::PVFullParallelScene*>(_fps);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::scene_parent
 *****************************************************************************/

PVParallelView::PVFullParallelScene const* PVParallelView::PVFullParallelViewSelectionRectangle::scene_parent() const
{
	return static_cast<PVParallelView::PVFullParallelScene*>(_fps);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::get_lines_view
 *****************************************************************************/

PVParallelView::PVLinesView& PVParallelView::PVFullParallelViewSelectionRectangle::get_lines_view()
{
	return scene_parent()->get_lines_view();
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::get_lines_view
 *****************************************************************************/

PVParallelView::PVLinesView const& PVParallelView::PVFullParallelViewSelectionRectangle::get_lines_view() const
{
	return scene_parent()->get_lines_view();
}

